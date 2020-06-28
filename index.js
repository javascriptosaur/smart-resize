const _cliProgress = require('cli-progress');
const Waifu2x = require('./Waifu2x');
const path = require('path');
const fs = require('fs');
const fse = require('fs-extra');

const encode = require('image-encode');
const { createCanvas, Image } = require('canvas');

const INPUT = path.join('input');
const OUTPUT = path.join('output');
const SCALE = +(process.env.SCALE) || 1.6;

const models = [require('./models/photo/noise3.json'), require('./models/art/scale2x.json')];
const waifu2x = new Waifu2x(models);

const generate = async (inputPath, outputPath) => {
    console.log(`${inputPath}`);

    const progressbar = new _cliProgress.Bar({}, _cliProgress.Presets.shades_classic);
    progressbar.start(100, 0);

    await waifu2x.init(inputPath);

    const canvas2x = createCanvas(waifu2x.width * 2, waifu2x.height * 2);
    const ctx = canvas2x.getContext('2d');

    waifu2x.on('progress', ({ pixels, progress, width, height, sx, sy, sw, sh, dx, dy }) => {
        progressbar.update(progress * 100);
        const pImg = new Image();
        pImg.src = Buffer.from(encode(pixels, [width, height], 'png'));
        ctx.drawImage(pImg, sx, sy, sw, sh, dx, dy, sw, sh);
    });

    await waifu2x.process();

    waifu2x.removeAllListeners('progress');

    const width2x = waifu2x.width * 2;
    const height2x = waifu2x.height * 2;

    //если есть альфа канал то надо его наложить
    if (waifu2x.alphaChannel) {
        const alphaImg = new Image();
        alphaImg.src = Buffer.from(encode(waifu2x.alphaChannel, [waifu2x.width, waifu2x.height], 'png'));
        const canvasAlpha = createCanvas(width2x, height2x);
        const ctxAlpha = canvasAlpha.getContext('2d');
        ctxAlpha.drawImage(alphaImg, 0, 0, waifu2x.width, waifu2x.height, 0, 0, width2x, height2x);
        const alphaImageData = ctxAlpha.getImageData(0, 0, width2x, height2x).data;

        var image = ctx.getImageData(0, 0, width2x, height2x);
        var imageData = image.data;

        for (let i = 3, n = imageData.length; i < n; i += 4) {
            imageData[i] = alphaImageData[i];
        }

        image.data = imageData;
        ctx.putImageData(image, 0, 0);
    }

    const canvas = createCanvas(waifu2x.width * SCALE, waifu2x.height * SCALE);
    canvas.getContext('2d').drawImage(canvas2x, 0, 0, width2x, height2x, 0, 0, waifu2x.width * SCALE, waifu2x.height * SCALE);

    const buf = canvas.toBuffer();
    fse.outputFileSync(outputPath, buf);

    progressbar.update(100);
    progressbar.stop();
};

//схлопывает многомерный массив в одномерный
const flatten = arr1 => arr1.reduce((acc, val) => (Array.isArray(val) ? acc.concat(flatten(val)) : acc.concat(val)), []);
//возвращает массив файлов в директории(рекурсивно)
const walkSync = d => (fs.statSync(d).isDirectory() ? fs.readdirSync(d).map(f => walkSync(path.join(d, f))) : d);

const inputFiles = flatten(walkSync(INPUT));
const outputFiles = inputFiles.map(p => path.relative(INPUT, p)).map(p => path.join(OUTPUT, p));

fse.emptyDirSync(OUTPUT);
(async _ => {
    for (let i = 0, n = inputFiles.length; i < n; i++) {
        await generate(path.join(__dirname, inputFiles[i]), path.join(__dirname, outputFiles[i].replace(/(\.[A-z]+)$/, '.png')));
    }
})();
