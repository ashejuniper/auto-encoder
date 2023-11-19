const { NeuralNetworkGPU } = require('brain.js');

/**
 * @typedef {Object} AutoEncoderTrainOptions
 * @property {number} errorThresh
 * Once the training error reaches `errorThresh`, training will be complete.
 * @property {number} iterations
 * Once the training epoch count reaches `iterations`, training will be
 * complete.
 * @property {number} learningRate
 * The rate at which values will be changed.
 */

/**
 * @typedef {import('brain.js/dist/lookup').ITrainingDatum[]} ITrainingData
 */

/**
 *
 * @param {string} word
 * The word to convert into a vector.
 * @param {number} wordLength
 * The maximum possible length of a word.
 * @returns {Float32Array}
 */
function word2vec (
    word,
    wordLength = 16
) {
    if (wordLength) {
        word = word.padEnd(wordLength);
    }

    const byteLength = wordLength * 4;
    const bitLength = byteLength * 8;

    const vec = new Float32Array(bitLength);

    let index = 0;

    for (let char of word) {
        let byte = char.charCodeAt(0);

        vec[index++] = byte & 0b0000_0001;
        vec[index++] = (byte & 0b0000_0010) >> 1;
        vec[index++] = (byte & 0b0000_0100) >> 2;
        vec[index++] = (byte & 0b0000_1000) >> 3;
        vec[index++] = (byte & 0b0001_0000) >> 4;
        vec[index++] = (byte & 0b0010_0000) >> 5;
        vec[index++] = (byte & 0b0100_0000) >> 6;
        vec[index++] = (byte & 0b1000_0000) >> 7;
    }

    return vec;
}

/**
 * Convert a vector of bits into a word.
 * @param {Float32Array} vec The vector of bits to convert into a word.
 * @returns {string} The decoded word.
 */
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

/**
 * @typedef {boolean[]|number[]|string} AutoDecodedData
 */

/**
 * @typedef {Float32Array} AutoEncodedData
 */

/**
 * @typedef {"boolean"|"number"|"string"} DataType
 */

/**
 * @typedef {Object} AutoEncoder
 */

/**
 * A type of neural network consisting of two subnetworks: an encoder, and a
 * decoder.
 * The encoder is responsible for converting the input into a smaller
 * representation via feature extraction.
 * The decoder is responsible for reconstructing the original input from a
 * vector of extracted features.
 *
 * Example usage:
 * ```
 * const autoEncoder = new AutoEncoder(10, 1, 'string');
 *
 * autoEncoder.train(["this", "is", "an", "example"]);
 *
 * const encoded = autoEncoder.encode("example");
 * const decoded = autoEncoder.decode(encoded);
 *
 * console.log(encoded, '->', decoded);
 * ```
 */
class AutoEncoder {
    /**
     * Create a new auto encoder.
     * @param {number} decodedDataSize
     * The size of the data prior to encoding, and after decoding.
     * @param {number} encodedDataSize
     * The size of the data after encoding, and prior to decoding.
     * @param {DataType} dataType
     * The type of data to encode.
     */
    constructor (
        decodedDataSize,
        encodedDataSize,
        dataType = 'number'
    ) {
        const transcodedDataSize
            = (
                encodedDataSize
                    + decodedDataSize
            )
                * 0.5
        ;

        /**
         * @type {DataType}
         */
        this._dataType = dataType;

        /**
         * @type {number}
         */
        this._encodedDataSize = encodedDataSize;

        /**
         * @type {number}
         */
        this._transcodedDataSize = transcodedDataSize;

        /**
         * @type {number}
         */
        this._decodedDataSize = decodedDataSize;

        /**
         * @type {NeuralNetworkGPU}
         */
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

        /**
         * @type {NeuralNetworkGPU}
         */
        this.decoder = new NeuralNetworkGPU(
            {
                hiddenLayers: [ this._getTranscodedDataSize() ],
                inputSize: this._getEncodedDataSize(),
                outputSize: this._getDecodedDataSize()
            }
        );
    }

    /**
     * Parse a stringified `AutoEncoder`.
     * @param {string} jsonString
     * A JSON string containing a stringified `AutoEncoder`.
     * @returns
     */
    static parse (jsonString) {
        const json = JSON.parse(jsonString);

        const autoEncoder = new AutoEncoder(
            json.decodedDataSize,
            json.encodedDataSize,
            json.dataType
        );

        autoEncoder.fromJSON(json);

        return autoEncoder;
    }

    /**
     * Decode encoded data.
     * @param {Float32Array} encodedData The encoded data to decode.
     * @returns {boolean[]|number[]|string} The decoded data.
     */
    decode (encodedData) {
        let decodedDataObject = this.decoder.run(encodedData);

        let decodedData = [];

        for (let i in decodedDataObject) {
            decodedData[i] = decodedDataObject[i];

            if (this._dataType === 'boolean') {
                decodedData[i] = decodedData[i] >= 0.5;
            }
        }

        if (this._dataType === 'string') {
            decodedData = vec2word(decodedData);
            decodedData = decodedData.substring(0, decodedData.indexOf(' '));
        }

        return decodedData;
    }

    /**
     * Encode data.
     * @param {AutoDecodedData} data
     * The data to encode.
     * @returns {AutoEncodedData}
     */
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

    /**
     * Load this `AutoEncoder`'s data from JSON.
     * @param {AutoEncoderJSON} json JSON representation of an `AutoEncoder`.
     */
    fromJSON (json) {
        if (typeof json === 'string') json = JSON.parse(json);

        this._decodedDataSize = json.decodedDataSize;
        this._transcodedDataSize = json.transcodedDataSize;
        this._encodedDataSize = json.encodedDataSize;

        this.encoder.fromJSON(json.encoder);
        this.decoder.fromJSON(json.decoder);
    }

    /**
     * Predict the decoded output of a given input data.
     * @param {AutoDecodedData} input
     * The input to predict the decoded output of.
     * @returns
     */
    run (input) {
        return this.decode(this.encode(input));
    }

    /**
     * Stringify this `AutoEncoder`.
     * @returns {string}
     * A JSON `string` containing this `AutoEncoder`.
     */
    stringify () {
        return JSON.stringify(this.toJSON());
    }

    /**
     *
     * @returns {object}
     * An object suitable for passing to `JSON.stringify()`.
     */
    toJSON () {
        return {
            encoder: this.encoder.toJSON(),
            decoder: this.decoder.toJSON()
        };
    }

    /**
     * Train the auto encoder on a training data set.
     * @param {ITrainingData} data
     * The data set to train the neural networks on.
     * @param {AutoEncoderTrainOptions} options
     * The options to pass to the neural network trainers.
     */
    train (data, options = {}) {
        this._trainEncoder(data, options);
        this._trainDecoder(data, options);
    }

    /**
     * Validate input by asserting that decoding the output of the encoder
     * reproduces the original input.
     * @param {AutoDecodedData} input
     * The input to validate.
     * @returns
     */
    validate (input) {
        const output = this.run(input);
        if (typeof output === 'string') return output === input;
        else throw new Error(`\`validate()\` not yet implemented for data type '${this._dataType}'.`);
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
        let size
            = (
                this._getEncodedDataSize()
                    + this._getDecodedDataSize()
            )
                * 0.5
        ;

        return Math.round(size);
    }

    _getVecSize () {
        return this._getWordSize() * 8;
    }

    _getWordSize () {
        return this._getDecodedDataSize() / 8;
    }

    _trainDecoder (data, options) {
        const trainingData = [];

        for (let output of data) {
            if (this._dataType === 'string') {
                output = output.padEnd(this._getWordSize());
            }

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

    _trainEncoder (data, options) {
        const trainingData = [];

        for (let input of data) {
            if (this._dataType === 'string') {
                input = input.padEnd(this._getWordSize());
            }

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
}

module.exports = AutoEncoder;
