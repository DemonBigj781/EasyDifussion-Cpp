/* Easy Diffusion C++ GIFs Plugin
 *        DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
 *                    Version 2, December 2004
 *
 * Copyright (C) 2022 Marc-Andre Ferland <madrang@gmail.com>
 *
 * Everyone is permitted to copy and distribute verbatim or modified
 * copies of this plugin, and changing it is allowed as long
 * as the name is changed.
 *
 *            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
 *   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
 *
 *  0. You just DO WHAT THE FUCK YOU WANT TO.
 *
 * @link http://www.wtfpl.net/
 * Derived from Madrang's SD-UI GIFs Plugin:
 * https://github.com/madrang/sd-ui-plugins/blob/beta/mads-gifs.plugin.js
 */
(function() { "use strict"
    const GITHUB_PAGE = "https://github.com/madrang/sd-ui-plugins"
    const VERSION = "3.0.0";
    const ID_PREFIX = "easy-diffusion-cpp-gifs";
    const GIF_SCRIPT = "/plugins/core/video_plugin/gif.js";
    const GIF_WORKER_SCRIPT = "/plugins/core/video_plugin/gif.worker.js";
    console.log('%s GIFs Version: %s', ID_PREFIX, VERSION);

    // Help and Community links
    const links = document.getElementById("community-links");
    (function() {
        if (links && !document.getElementById(`${ID_PREFIX}-link`)) {
            // Add link to plugin repo.
            const pluginLink = document.createElement('li');
            pluginLink.innerHTML = `<a id="${ID_PREFIX}-link" href="${GITHUB_PAGE}" target="_blank"><i class="fa-solid fa-code-merge"></i> GIF plugin source</a>`;
            links.appendChild(pluginLink);
        }
    })();

    function copyImg(ctx, src, dest={}) {
        return new Promise(function(resolve, reject) {
            const drawImage = function(image) {
                ctx.drawImage(image
                    , src.x || 0, src.y || 0
                    , src.w || src.width || image.naturalWidth
                    , src.h || src.height || image.naturalHeight
                    , dest.x || 0, dest.y || 0
                    , dest.w || dest.width || ctx.width || ctx.canvas.width
                    , dest.h || dest.height || ctx.height || ctx.canvas.height
                );
                resolve(image);
            }
            try {
                if ((typeof HTMLImageElement !== "undefined" && src.data instanceof HTMLImageElement)
                    || (typeof SVGImageElement !== "undefined" && src.data instanceof SVGImageElement)
                    || (typeof HTMLCanvasElement !== "undefined" && src.data instanceof HTMLCanvasElement)
                    || (typeof ImageBitmap !== "undefined" && src.data instanceof ImageBitmap)
                ) {
                    drawImage(src.data);
                    return;
                }
                const image = new Image();
                image.addEventListener('load', () => drawImage(image));
                image.addEventListener('error', () => reject(new Error('Failed to load image.')));
                image.src = src.url || src.data;
                if (!image.src) {
                    reject(new Error("GIF frame has no image data."));
                }
            } catch (e) {
                reject(e);
            }
        });
    }

    function createCanvas(width, height) {
        if (typeof OffscreenCanvas === "function") {
            return new OffscreenCanvas(width, height);
        }
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }

    function blobToDataURL(blob) {
        return new Promise(function(resolve, reject) {
            const reader = new FileReader();
            reader.addEventListener("load", () => resolve(reader.result));
            reader.addEventListener("error", () => reject(reader.error || new Error("Failed to read GIF.")));
            reader.readAsDataURL(blob);
        });
    }

    function createGif(width, height) {
        return new GIF({
            workerScript: GIF_WORKER_SCRIPT,
            workers: Math.max(1, Math.min(2, navigator.hardwareConcurrency || 2)),
            quality: 10,
            width,
            height,
        });
    }

    function renderGif(gif) {
        if (gif.frames.length === 0) {
            return Promise.reject(new Error("No frames were generated for the GIF."));
        }
        return new Promise(function(resolve, reject) {
            gif.on("finished", (blob) => blobToDataURL(blob).then(resolve, reject));
            gif.on("abort", () => reject(new DOMException("GIF encoding was cancelled.", "AbortError")));
            try {
                gif.render();
            } catch (error) {
                reject(error);
            }
        });
    }

    function gifResult(data, sourceResult) {
        const sourceOutput = sourceResult?.output?.[0] || {};
        return {
            status: "succeeded",
            output: [{
                seed: sourceOutput.seed,
                data,
            }],
        };
    }

    function throwIfAborted(signal) {
        if (signal.aborted) {
            throw new DOMException("GIF generation was cancelled.", "AbortError");
        }
    }

    function writeImg(ctx, header, img, transparency, disposalMethod) {
        const ct = img.lctFlag ? img.lct : header.gct; // TODO: What if neither exists?
        const cData = ctx.getImageData(img.leftPos, img.topPos, img.width, img.height);

        img.pixels.forEach(function(pixel, i) {
            // cData.data === [R,G,B,A,...]
            if (transparency !== pixel) { // This includes null, if no transparency was defined.
                cData.data[i * 4 + 0] = ct[pixel][0];
                cData.data[i * 4 + 1] = ct[pixel][1];
                cData.data[i * 4 + 2] = ct[pixel][2];
                cData.data[i * 4 + 3] = 255; // Opaque.
            } else {
                // TODO: Handle disposal method properly.
                // XXX: When I get to an Internet connection, check which disposal method is which.
                if (disposalMethod === 2 || disposalMethod === 3) {
                    cData.data[i * 4 + 3] = 0; // Transparent.
                    // XXX: This is very very wrong.
                } else {
                    // disposalMethod should be null (no GCE), 0, or 1; leave the pixel as it is.
                    // assert(disposalMethod === null || disposalMethod === 0 || disposalMethod === 1);
                    // XXX: If this is the first frame (and we *do* have a GCE),
                    // disposalMethod will be null, but we want to set undefined
                    // pixels to the background color.
                }
            }
        });
        ctx.putImageData(cData, img.leftPos, img.topPos);
    };

    const GIF_DISPOSAL_METHODS = {
        notSpecified: 0
        , doNotDispose: 1
        , restoreToBackgroundColor: 2
        , restoreToPrevious: 3
    };
    Object.freeze(GIF_DISPOSAL_METHODS);
    loadScript(GIF_SCRIPT).then(function() {
        if (!('OUTPUTS_FORMATS' in PLUGINS)) {
            return;
        }
        PLUGINS['OUTPUTS_FORMATS'].register(function gif() {
            return (reqBody) => {
                const renderRequest = Object.assign({}, reqBody, {
                    output_format: "png",
                    stream_image_progress: true,
                    stream_image_progress_interval: 1,
                });
                const instance = new SD.RenderTask(renderRequest);
                const enqueue = instance.enqueue;
                const abort = instance.abort;
                const gif = createGif(reqBody.width, reqBody.height);
                const offscreenOutput = createCanvas(reqBody.width, reqBody.height);
                const outputCtx = offscreenOutput.getContext("2d");
                let frameCount = 0;
                let encoding = false;

                instance.enqueue = function(callback) {
                    return enqueue.call(this, async function(event) {
                        if (typeof event?.update?.output === "object" && !encoding) {
                            const updateOutput = event.update.output;
                            const src = {};
                            if (updateOutput[0]?.path) {
                                src.url = updateOutput[0].path + "?t=" + Date.now();
                            }
                            if (updateOutput[0]?.data) {
                                src.data = updateOutput[0].data;
                            }
                            if (src.url || src.data) {
                                await copyImg(outputCtx, src);
                                gif.addFrame(outputCtx, {copy: true, delay: 125});
                                frameCount++;
                            }
                        }
                        return await Promise.resolve(callback?.call(this, event));
                    }).then(async function(result) {
                        if (frameCount === 0 && (result?.output?.[0]?.data || result?.output?.[0]?.path)) {
                            await copyImg(outputCtx, {
                                data: result.output[0].data,
                                url: result.output[0].path,
                            });
                            gif.addFrame(outputCtx, {copy: true, delay: 500});
                            frameCount++;
                        } else if (frameCount > 0) {
                            // Hold the finished image briefly before the animation loops.
                            gif.addFrame(outputCtx, {copy: true, delay: 500});
                        }
                        encoding = true;
                        return gifResult(await renderGif(gif), result);
                    });
                };
                instance.abort = function(reason) {
                    if (gif.running) {
                        gif.abort();
                    }
                    return abort.call(this, reason);
                };
                return instance;
            }
        });
        PLUGINS['OUTPUTS_FORMATS'].register(function stepAnim() {
            const processImage = async function*(reqBody, callback, signal) {
                console.log(`GIF - Starting ${reqBody.width}x${reqBody.height} gif render.`);

                const gif = createGif(reqBody.width, reqBody.height);
                const offscreenOutput = createCanvas(reqBody.width, reqBody.height);
                const outputCtx = offscreenOutput.getContext("2d");
                outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                const renderFrames = [];

                let delay = 250;
                const advance_step = 0.088;
                for (let num_inference_steps = 5;
                    num_inference_steps <= reqBody.num_inference_steps;
                    num_inference_steps += Math.max(1, Math.floor(num_inference_steps * advance_step))
                ) {
                    throwIfAborted(signal);
                    console.log(`Gif.frame Starting Render ${num_inference_steps}/${reqBody.num_inference_steps}`);
                    const result = yield SD.render(Object.assign({}, reqBody, {
                        num_inference_steps
                        , output_format: 'png'
                    }), callback);
                    console.log('Gif.frame Render response %o', result);

                    const outputData = result?.output?.[0]?.data;
                    if (!outputData) {
                        throw new Error("Step animation render returned no image data.");
                    }
                    renderFrames.push(outputData);

                    // Clear output buffer
                    outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                    // Read back result.
                    const img = yield copyImg(outputCtx, {data: outputData, width: reqBody.width, height: reqBody.height});
                    // Add to gif renderer.
                    gif.addFrame(outputCtx, {copy: true, delay});
                    console.log('Added new frame %o to gif %o', img, gif);
                }
                // Reverse animation and add frames again.
                renderFrames.reverse();
                for(const imgData of renderFrames) {
                    throwIfAborted(signal);
                    // Clear output buffer
                    outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                    // Read back result.
                    const img = yield copyImg(outputCtx, {data: imgData, width: reqBody.width, height: reqBody.height});
                    // Add to gif renderer.
                    gif.addFrame(outputCtx, {copy: true, delay});
                    console.log('Added new frame %o to gif %o', img, gif);
                }

                // Start final render
                const gifDataUrl = await renderGif(gif);
                return gifResult(gifDataUrl);
            };
            return (reqBody) => {
                const controller = new AbortController();
                return {
                    abort: () => controller.abort()
                    , enqueue: function(callback) {
                        const process = processImage(reqBody, callback, controller.signal);
                        return SD.Task.enqueue(process);
                    }
                };
            }
        });
        PLUGINS['OUTPUTS_FORMATS'].register(function morph() {
            const parsePrompt = function(text, params) {
                for (const [argName, argValue] of Object.entries(params)) {
                    text = text.replace(new RegExp(`{${argName}}`, "igm"), argValue.toFixed(3))
                }
                return text;
            };
            const processImage = async function*(reqBody, callback, signal) {
                console.log(`GIF - Starting ${reqBody.width}x${reqBody.height} gif render.`);

                const gif = createGif(reqBody.width, reqBody.height);
                const offscreenOutput = createCanvas(reqBody.width, reqBody.height);
                const outputCtx = offscreenOutput.getContext("2d");
                outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                const renderFrames = [];

                let delay = 99; // playback time per frame in milliseconds.
                const advance_step = 2;
                const rangeStart = 1;
                const rangeEnd = 99;
                for (let weight_step = 0; rangeStart + weight_step <= rangeEnd; weight_step += advance_step) {
                    throwIfAborted(signal);
                    let promptOptions;
                    if (false) {
                        const blendAlpha = (2.0 / (1.0 + Math.exp(-0.05 * weight_step))) - 1.0;
                        promptOptions = { x: 100 * (1.0 - blendAlpha), y: 100 * blendAlpha };
                    } else {
                        promptOptions = { x: rangeEnd - weight_step, y: rangeStart + weight_step };
                    }
                    console.log(`Gif.frame Starting Render ${weight_step / advance_step}/${Math.floor((1 + rangeEnd - rangeStart) / advance_step)} using options %o`, promptOptions);
                    const result = yield SD.render(Object.assign({}, reqBody, {
                        prompt: parsePrompt(reqBody.prompt, promptOptions)
                        , output_format: 'png'
                    }), callback);
                    console.log('Gif.frame Render response %o', result);

                    const outputData = result?.output?.[0]?.data;
                    if (!outputData) {
                        throw new Error("Morph animation render returned no image data.");
                    }
                    renderFrames.push(outputData);

                    // Clear output buffer
                    outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                    // Read back result.
                    const img = yield copyImg(outputCtx, {data: outputData, width: reqBody.width, height: reqBody.height});
                    // Add to gif renderer.
                    gif.addFrame(outputCtx, {copy: true, delay});
                    console.log('Added new frame %o to gif %o', img, gif);
                }
                // Reverse animation and add frames again.
                renderFrames.reverse();
                for(const imgData of renderFrames) {
                    throwIfAborted(signal);
                    // Clear output buffer
                    outputCtx.clearRect(0, 0, reqBody.width, reqBody.height);
                    // Read back result.
                    const img = yield copyImg(outputCtx, {data: imgData, width: reqBody.width, height: reqBody.height});
                    // Add to gif renderer.
                    gif.addFrame(outputCtx, {copy: true, delay});
                    console.log('Added new frame %o to gif %o', img, gif);
                }

                // Start final render
                const gifDataUrl = await renderGif(gif);
                return gifResult(gifDataUrl);
            };
            return (reqBody) => {
                const controller = new AbortController();
                return {
                    abort: () => controller.abort()
                    , enqueue: function(callback) {
                        const process = processImage(reqBody, callback, controller.signal);
                        return SD.Task.enqueue(process);
                    }
                };
            }
        });

        const GIF_HEADER = 'data:image/gif;base64,'
        PLUGINS['TASK_CREATE'].push(function(event) {
            if (typeof event?.reqBody?.init_image !== 'string' || !event.reqBody.init_image.startsWith(GIF_HEADER)) {
                return
            }
            const data = event.reqBody.init_image.slice(GIF_HEADER.length);
            const decodedData = (typeof atob === 'function' ? atob(data) : Buffer.from(data, 'base64'));
            const stream = new Stream(decodedData);
            const imgArr = [];
            const eventsArr = [];
            let imgHeader = undefined;
            const readGifPromiseSrc = new PromiseSource();
            const processImage = async function*(callback, signal) {
                const inputCanvas = document.createElement('canvas');
                inputCanvas.width = imgHeader.width;
                inputCanvas.height = imgHeader.height;
                const inputCtx = inputCanvas.getContext("2d");
                inputCtx.clearRect(0, 0, event.reqBody.width, event.reqBody.height);

                yield readGifPromiseSrc.promise;
                console.log(`GIF2GIF - Starting ${imgHeader.width}x${imgHeader.height} gif render of ${imgArr.length} frames.`);

                const gif = createGif(event.reqBody.width, event.reqBody.height);
                const offscreenOutput = createCanvas(event.reqBody.width, event.reqBody.height);
                const outputCtx = offscreenOutput.getContext("2d");
                outputCtx.clearRect(0, 0, event.reqBody.width, event.reqBody.height);

                let transparency = undefined;
                let disposalMethod = undefined;
                let delay = 500;
                let lastFrameData = undefined;
                for (let srcImg of eventsArr) {
                    throwIfAborted(signal);
                    if (srcImg.type === 'ext') {
                        // pixel manipulation events...
                        if (srcImg.extType === 'gce') {
                            transparency = srcImg.transparencyGiven ? srcImg.transparencyIndex : null;
                            if (srcImg.delayTime) {
                                // GIF graphic-control delays are stored in 1/100 second units.
                                delay = Math.max(20, srcImg.delayTime * 10);
                            }
                            disposalMethod = srcImg.disposalMethod;
                            if (disposalMethod == GIF_DISPOSAL_METHODS.restoreToBackgroundColor) {
                                inputCtx.clearRect(0, 0, imgHeader.width, imgHeader.height);
                            } else if (disposalMethod == GIF_DISPOSAL_METHODS.restoreToPrevious) {
                                inputCtx.globalAlpha = 1;
                                yield copyImg(inputCtx, {data: lastFrameData, width: imgHeader.width, height: imgHeader.height});
                            } else if (disposalMethod == GIF_DISPOSAL_METHODS.doNotDispose) {
                                lastFrameData = inputCanvas.toDataURL("image/png")
                            }
                        }
                        continue;
                    }

                    // keep some of the last output in the frame to stabilise Stable Diffusion.
                    //inputCtx.globalAlpha = prompt_strength / 2.0;
                    //inputCtx.drawImage(offscreenOutput
                    //    , 0, 0, offscreenOutput.width, offscreenOutput.height // Src
                    //    , 0, 0, imgHeader.width, imgHeader.height // Dest
                    //);

                    // Write the updated pixels of the current gif frame to the ctx.
                    inputCtx.globalAlpha = 1;
                    writeImg(inputCtx, imgHeader, srcImg, transparency, disposalMethod);

                    // Send to backend
                    console.log(`Gif.frame Starting Render ${imgArr.indexOf(srcImg) + 1} of ${imgArr.length}`);
                    const result = yield SD.render(Object.assign({}, event.reqBody, {
                        init_image: inputCanvas.toDataURL("image/png")
                        , output_format: 'png'
                    }), callback);
                    console.log('Gif.frame Render response %o', result);

                    // Clear output buffer
                    outputCtx.clearRect(0, 0, event.reqBody.width, event.reqBody.height);
                    // Read back result.
                    const outputData = result?.output?.[0]?.data;
                    if (!outputData) {
                        throw new Error("GIF-to-GIF render returned no image data.");
                    }
                    const img = yield copyImg(outputCtx, {data: outputData});
                    // Add to gif renderer.
                    gif.addFrame(outputCtx, {copy: true, delay});
                    console.log('Added new frame %o to gif %o', img, gif);

                }
                // Start final render
                const gifDataUrl = await renderGif(gif);
                return gifResult(gifDataUrl);
            };
            parseGIF(stream, {
                // Print header to console.
                hdr: (h) => {
                    imgHeader = h;
                    console.log(h);
                }
                // Get Image data.
                , img: (imgData) => {
                    console.log(imgData);
                    eventsArr.push(imgData);
                    imgArr.push(imgData);
                }
                , gce: function(gce) {
                    console.log(gce);
                    eventsArr.push(gce);
                }
                // End of file reached.
                , eof: () => {
                    readGifPromiseSrc.resolve(eventsArr);
                    console.log('GIF read completed! ImageData: %o', eventsArr);
                }
            });
            const controller = new AbortController();
            event.instance = {
                abort: () => controller.abort()
                , enqueue: function(callback) {
                    const process = processImage(callback, controller.signal);
                    return SD.Task.enqueue(process);
                }
            };
            event.reqBody.output_format = 'gif';
        })
    }, (reason) => console.error(reason));

    // Register selftests when loaded by jasmine.
    if (typeof PLUGINS?.SELFTEST === 'object') {
        PLUGINS.SELFTEST[ID_PREFIX + " render tasks"] = function() {
            it('should be able to run a test...', function() {
                expect(function() {
                    SD.sessionId = undefined
                }).toThrowError("Can't set sessionId to undefined.")
            })
        }
    }
})();
