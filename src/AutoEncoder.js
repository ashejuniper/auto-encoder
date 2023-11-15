const { NeuralNetworkGPU } = require('brain.js');

function word2vec (
    word,
    wordLength = null
) {
    if (wordLength) {
        word = word.padEnd(wordLength);
    }

    const vec = [];

    for (let char of word) {
        let byte = char.charCodeAt(0);

        let bit0 = byte & 0b0000_0001;
        let bit1 = (byte & 0b0000_0010) >> 1;
        let bit2 = (byte & 0b0000_0100) >> 2;
        let bit3 = (byte & 0b0000_1000) >> 3;
        let bit4 = (byte & 0b0001_0000) >> 4;
        let bit5 = (byte & 0b0010_0000) >> 5;
        let bit6 = (byte & 0b0100_0000) >> 6;
        let bit7 = (byte & 0b1000_0000) >> 7;

        vec.push(bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7);
    }

    return vec;
}

function vec2word (
    vec
) {
    const bytes = [];

    for (
        let vecIndex = 0;
        vecIndex < vec.length;
        vecIndex += 8
    ) {
        let byte = 0x00;

        for (
            let localBitIndex = 0;
            localBitIndex < 8;
            localBitIndex++
        ) {
            const bitIndex = vecIndex + localBitIndex;
            const predictedBit = vec[bitIndex];

            const bit = Math.round(predictedBit);

            byte |= bit << localBitIndex;
        }

        bytes.push(byte);
    }

    let word = String.fromCharCode(...bytes).trim();

    return word;
}

class AutoEncoder {
    constructor (
        decodedDataSize,
        encodedDataSize,
        dataType = 'number[]'
    ) {
        const transcodedDataSize
            = (
                encodedDataSize
                    + decodedDataSize
            )
                * 0.5
        ;

        this._dataType = dataType;

        this._encodedDataSize = encodedDataSize;
        this._transcodedDataSize = transcodedDataSize;
        this._decodedDataSize = decodedDataSize;

        this.encoder = new NeuralNetworkGPU(
            {
                hiddenLayers: [
                    this._getTranscodedDataSize(),
                    this._getEncodedDataSize(),
                    this._getTranscodedDataSize()
                ],
                inputSize: this._getDecodedDataSize(),
                outputSize: this._getDecodedDataSize()
            }
        );

        this.decoder = new NeuralNetworkGPU(
            {
                hiddenLayers: [ this._getTranscodedDataSize() ],
                inputSize: this._getEncodedDataSize(),
                outputSize: this._getDecodedDataSize()
            }
        );
    }

    encode (data) {
        if (this._dataType === 'string') {
            if (data.length < this._getWordSize()) {
                data.padEnd(this._getWordSize());
            }

            data = word2vec(
                data,
                this._getWordSize()
            );
        }

        this.encoder.run(data);

        const encodedDataLayer = this.encoder.outputs[2];

        let encodedData = encodedDataLayer.toArray();

        return encodedData;
    }

    decode (encodedData) {
        let decodedDataObject = this.decoder.run(encodedData);

        let decodedData = [];

        for (let i in decodedDataObject) {
            decodedData[i] = decodedDataObject[i];
        }

        if (this._dataType === 'string') {
            decodedData = vec2word(decodedData).trim();
        }

        return decodedData;
    }

    toJSON () {
        return {
            encoder: this.encoder.toJSON(),
            decoder: this.decoder.toJSON()
        };
    }

    train (data, options) {
        this._trainEncoder(data, options);
        this._trainDecoder(data, options);
    }

    _getDecodedDataSize () {
        let size = this._decodedDataSize;

        if (this._dataType === 'string') {
            size *= 8;
        }

        return size;
    }

    _getEncodedDataSize () {
        let size = this._encodedDataSize;

        if (this._dataType === 'string') {
            size *= 8;
        }

        return size;
    }

    _getTranscodedDataSize () {
        let size = this._transcodedDataSize;

        if (this._dataType === 'string') {
            size *= 8;
        }

        return size;
    }

    _getWordSize () {
        return this._getDecodedDataSize() / 8;
    }

    _getVecSize () {
        return this._getWordSize() * 8;
    }

    _trainEncoder (data, options) {
        const trainingData = [];

        for (let input of data) {
            input = input.padEnd(this._getWordSize());

            if (typeof input === 'string') {
                input = word2vec(
                    input,
                    this._getWordSize()
                );

                this._dataType = 'string';
            }

            let output = input;

            if (typeof output === 'string') {
                output = output.padEnd(this._getWordSize());

                output = word2vec(
                    output,
                    this._getWordSize()
                );

                this._dataType = 'string';
            }

            const entry = {
                input,
                output
            };

            trainingData.push(entry);
        }

        this.encoder.train(trainingData, options);
    }

    _trainDecoder (data, options) {
        const trainingData = [];

        for (let output of data) {
            output = output.padEnd(this._getWordSize());

            const rawOutput = output;

            if (typeof output === 'string') {
                output = word2vec(
                    rawOutput,
                    this._getWordSize()
                );

                this._dataType = 'string';
            }
            const input = this.encode(rawOutput);

            const entry = {
                input,
                output
            };

            trainingData.push(entry);
        }

        this.decoder.train(trainingData, options);
    }

    _createDecoder () {
        this.decoder = new NeuralNetworkGPU(
            {
                hiddenLayers: [ this._getTranscodedDataSize() ],
                inputSize: this._getEncodedDataSize(),
                outputSize: this._getDecodedDataSize()
            }
        );
    }
}

module.exports = AutoEncoder;
