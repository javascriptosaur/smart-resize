const tf = require('./extend');
const imageGet = require('get-image-data');
const EventEmitter = require('events');

class Waifu2x extends EventEmitter {
    constructor(params) {
        super();
        this.params = params;
    }

    //загружает и подготавливает картинку
    async init(filename) {
        this.pixels = undefined;
        this.alphaChannel = undefined;
        this.height = 0;
        this.width = 0;

        return new Promise((res, rej) => {
            imageGet(filename, (err, info) => {
                if (err) {
                    rej(err);
                } else {
                    const { width, height, data } = info;
                    const numChannels = 3;
                    const numPixels = width * height;
                    const pixels = new Int32Array(numPixels * numChannels);
                    const alphaChannel = new Int32Array(numPixels);

                    for (let i = 0; i < numPixels; i++) {
                        for (let channel = 0; channel < numChannels; ++channel) {
                            pixels[i * numChannels + channel] = data[i * 4 + channel];
                        }
                        alphaChannel[i] = data[i * 4 + 3];
                    }

                    this.pixels = pixels;
                    this.alphaChannel = alphaChannel;
                    this.height = height;
                    this.width = width;

                    //если нет прозрачных пикселей то удалить alphaChannel
                    if (alphaChannel.reduce((a, b) => (b === 255 ? a + 1 : a), 0) === alphaChannel.length) {
                        this.alphaChannel = undefined;
                    } else {
                        this.alphaChannel = this.alphaChannel.reduce((accum, current) => {
                            accum.push(0, 0, 0, current);
                            return accum;
                        }, []);
                    }

                    res();
                }
            });
        });
    }

    //возращает тензор из куска картинки
    _getTensorFromRect(x, y, width, height) {
        const numChannels = 3;
        const numPixels = width * height;
        const pixels = new Int32Array(numPixels * numChannels);

        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                let py = y + i;
                let px = x + j;
                py < 0 && (py = 0);
                py > this.height-1 && (py = this.height-1);
                px < 0 && (px = 0);
                px > this.width-1 && (px = this.width-1);
                const ip = ((py) * this.width + px) * numChannels;
                const op = (i * height + j) * numChannels;
                for (let channel = 0; channel < numChannels; ++channel) {
                    
                    pixels[op + channel] = this.pixels[ip + channel];
                    
                }
            }
        }

        const outShape = [height, width, numChannels];
        return tf.tensor3d(pixels, outShape, 'int32');
    }

    async process() {
        const max_size = 16;
        const margin = 6;

        for (let i = 0; i * max_size < this.height; i++) {
            let s_y = i * max_size;
            let m_y = s_y - margin;
            let s_h = max_size;
            let m_h = s_h + margin * 2;
            if ((i + 1) * max_size - this.height > 0) {
                s_h = this.height - s_y;
            }
            for (let j = 0; j * max_size < this.width; j++) {
                let s_x = j * max_size;
                let m_x = s_x - margin;
                let s_w = max_size;
                let m_w = s_w + margin * 2;
                if ((j + 1) * max_size - this.width > 0) {
                    s_w = this.width - s_x;
                }

                let im = this._getTensorFromRect(m_x, m_y, m_w, m_h);
                im = await this.generate(im, 0, true);
                im = await this.generate(im, 1, false);

                const pixels = await tf.browser.toPixels(im);

                this.emit('progress', {
                    pixels,
                    progress: (i * max_size) / this.height,
                    width: im.shape[1],
                    height: im.shape[0],
                    sx: margin * 2,
                    sy: margin * 2,
                    sw: s_w * 2,
                    sh: s_h * 2,
                    dx: s_x * 2,
                    dy: s_y * 2
                });
            }
        }

        //this.tidy();
    }

    _loadModel(input_shape, param_id) {
        const model = tf.sequential();
        model.add(
            tf.layers.conv2d({
                filters: this.params[param_id][0]['nOutputPlane'],
                kernelSize: [this.params[param_id][0]['kH'], this.params[param_id][0]['kW']],
                kernelInitializer: 'zeros',
                padding: 'same',
                weights: [tf.tensor(this.params[param_id][0]['weight']).transpose([2, 3, 1, 0]), tf.tensor(this.params[param_id][0]['bias'])],
                useBias: true,
                inputShape: input_shape,
                dataFormat: 'channelsFirst'
            })
        );
        model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
        for (let i in this.params[param_id]) {
            if (i == 0) continue; // i is 'string' type, use '==' but not '==='
            const param = this.params[param_id][i];
            model.add(
                tf.layers.conv2d({
                    filters: param['nOutputPlane'],
                    kernelSize: [param['kH'], param['kW']],
                    kernelInitializer: 'zeros',
                    padding: 'same',
                    weights: [tf.tensor(param['weight']).transpose([2, 3, 1, 0]), tf.tensor(param['bias'])],
                    useBias: true,
                    dataFormat: 'channelsFirst'
                })
            );
            model.add(tf.layers.leakyReLU({ alpha: 0.1 }));
        }
        return model;
    }

    async _loadImageY(im, is_noise) {
        im = tf.tidy(() => {
            let _im = im.convert('YCbCr');
            tf.dispose(im);
            return _im;
        });

        im = tf.tidy(() => {
            let _im;
            if (is_noise) {
                _im = im.asType('float32');
            } else {
                _im = tf.image.resizeNearestNeighbor(im, [im.shape[0] * 2, im.shape[1] * 2]).asType('float32');
            }
            tf.dispose(im);
            return _im;
        });

        let x = tf.tidy(() => {
            return im
                .slice([0, 0, 0], [im.shape[0], im.shape[1], 1])
                .reshape([1, 1, im.shape[0], im.shape[1]])
                .div(tf.scalar(255.0));
        });

        return [im, x];
    }

    async _loadImageRGB(im, is_noise) {
        im = tf.tidy(() => {
            let _im;
            if (is_noise) {
                _im = im.asType('float32');
            } else {
                _im = tf.image.resizeNearestNeighbor(im, [im.shape[0] * 2, im.shape[1] * 2]).asType('float32');
            }
            tf.dispose(im);
            return _im;
        });

        let x = tf.tidy(() => {
            let r = im.slice([0, 0, 0], [im.shape[0], im.shape[1], 1]);
            let g = im.slice([0, 0, 1], [im.shape[0], im.shape[1], 1]);
            let b = im.slice([0, 0, 2], [im.shape[0], im.shape[1], 1]);
            return tf
                .stack([r, g, b])
                .reshape([1, 3, im.shape[0], im.shape[1]])
                .div(tf.scalar(255.0));
        });

        return [im, x];
    }

    async generate(tensor, param_id = 0, is_noise = false) {
        const input_channel = this.params[param_id][0]['nInputPlane'];

        if (!this.models) {
            this.models = [];
            for (let i in this.params) {
                const t_input_channel = this.params[i][0]['nInputPlane'];
                this.models.push(this._loadModel([t_input_channel, null, null], i));
            }
        }
        let model = this.models[param_id];

        let im, x;

        if (input_channel === 1) {
            [im, x] = await this._loadImageY(tensor, is_noise);
        } else {
            [im, x] = await this._loadImageRGB(tensor, is_noise);
        }

        im = tf.tidy(() => {
            let _im = im;
            let y = model.predict(x);

            if (input_channel === 1) {
                y = y
                    .mul(tf.scalar(255.0))
                    .clipByValue(0, 255)
                    .reshape([im.shape[0], im.shape[1], 1]);
                let cb = im.slice([0, 0, 1], [im.shape[0], im.shape[1], 1]);
                let cr = im.slice([0, 0, 2], [im.shape[0], im.shape[1], 1]);
                im = tf.stack([y, cb, cr], -1).reshape([im.shape[0], im.shape[1], 3]);
                im = im
                    .setMode('YCbCr')
                    .convert('RGB')
                    .asType('int32');
            } else {
                im = y.mul(tf.scalar(255.0)).clipByValue(0, 255);
                let r = im.slice([0, 0, 0, 0], [1, 1, im.shape[2], im.shape[3]]);
                let g = im.slice([0, 1, 0, 0], [1, 1, im.shape[2], im.shape[3]]);
                let b = im.slice([0, 2, 0, 0], [1, 1, im.shape[2], im.shape[3]]);
                im = tf.stack([r, g, b], -1).reshape([im.shape[2], im.shape[3], 3]);
                im = im.setMode('RGB').asType('int32');
            }
            tf.dispose(_im);
            return im;
        });

        return im;
    }

    tidy() {
        for (let model of this.models) {
            for (let layer of model.layers) {
                tf.dispose(layer.getWeights());
            }
        }
    }
}

module.exports = Waifu2x;
