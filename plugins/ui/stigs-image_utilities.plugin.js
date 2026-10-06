/**
 * Image Utilities
 * v.1.13, last updated: 02/10/2024
 * By The Stig
 * 
 * 
 * Change Log
 * 02/10/2024 Added facility to save file prefixed with prompt
 *
 * Free to use with the CMDR2 Stable Diffusion UI.
 *  
 */

(function() { "use strict"

	const VERSION = "1.13";
	const ID_PREFIX = "TheStig-Image Utilities";
	console.log('%s The Stig - Image Utilities Version: %s', ID_PREFIX, VERSION);
	
	const style = document.createElement('style');
    style.textContent = `#${ID_PREFIX}-webcamButton {margin-top: 8px;}`;
	
	document.head.append(style);
    const editorInputs = document.getElementById("editor-inputs");

    const buttonsContainer = document.createElement('div');
    buttonsContainer.id = `${ID_PREFIX}-webcamContainer`;
    editorInputs?.appendChild(buttonsContainer);
	
	
	var useDefaultFname = false;		// Set to True to revert to default file name convention
	var strDefaultFname = "ED";
	var strDefaultCnetFname = "CNet";
	var maxFnameChar = 20;				// Number of characters to use from prompt in filename

	
	var imgUtilsSettings = {
		imgUtils1: 'fa-solid fa-undo', 					// Rotate 90deg CCW
		imgUtils2: 'fa-solid fa-redo', 					// Rotate 90deg CW
		imgUtils3: 'fa-solid fa-puzzle-piece', 			// Tile
		imgUtils4: 'fa-solid fa-mask', 					// Invert Mask
		imgUtils5: 'fa-solid fa-file-export', 			// Download
		imgUtils6: 'fa-solid fa-info-circle'			// Show Info
	};
		
	PLUGINS['IMAGE_INFO_BUTTONS'].push([
		{ html: '<span class="imageadjust-label" style="background-color:transparent;background: rgba(0,0,0,0.5)">Image Utils:</span>', type: 'label', on_click: onImageAdjustClick},
		{ html: '<div title = "Rotate 90deg CCW"><i class="' + imgUtilsSettings.imgUtils1 + '"></i></div>', on_click: onRotate90CCW},
		{ html: '<div title = "Rotate 90deg CW"><i class="' + imgUtilsSettings.imgUtils2 + '"></i></div>', on_click: onRotate90CW},
		{ html: '<div title = "Tile Image"><i class="' + imgUtilsSettings.imgUtils3 + '"></i></div>', on_click: onTileImage},
		//{ html: '<div title = "Use Mask"><i class="' + imgUtilsSettings.imgUtils4 + '"></i></div>', on_click: onSaveMask},  // onInvertMask fa-stripe-s
		{ html: '<div title = "Download image and MetaData"><i class="' + imgUtilsSettings.imgUtils5 + '"></i></div>', on_click: onDownloadMeta},
		{ html: '<div title = "Show/Hide info"><i class="' + imgUtilsSettings.imgUtils6 + '"></i></div>', on_click: onDisplayInfo},	
	])
	
	
	
	

	function buildRequest(options = {}) {
        const newTaskRequest = modifyCurrentRequest(getCurrentUserRequest().reqBody, { //TODO remove getCurrentUserRequest after is fixed upstream.
            session_id: sessionId
        });
		if ('prompt' in options) {
            newTaskRequest.reqBody.prompt = options.prompt;
        } else {
            newTaskRequest.reqBody.prompt = 'Test';
        }
        return newTaskRequest;
    }
	
	function getStartNewTaskHandler() {
        return async function(event) {
            //const options = await showPopup(mode, reqBody);
            //if (options.cancelled) {
            //    return;
            //}
            const options = {};
            const newTaskRequest = buildRequest(options);
            createTask(newTaskRequest);
            initialText.style.display = 'none';
        }
    }

	function onDownloadMeta(origRequest, image) {
		//console.log('**Downloading with Meta Data**');
		var dImage = makeImageDrawable(image);  // convert DOM img to canvas
		var ftype=document.getElementById("output_format").value;
		//console.log(image);
		const newImage = document.createElement('img');
		newImage.replaceWith(image);
		newImage.setAttribute("Comments",origRequest);
		newImage.setAttribute("Comments",origRequest);
		//console.log(ftype);
		//console.log(newImage);
		var fname=null;
		var txtname=null;
		const now = new Date();
		const isoDate = now.toISOString();
		const filename = isoDate.replace(/:/g,"");
		const downloadname=filename.replace(/-/g,"");
		const download=downloadname.replace(/T/g,"-");
		//console.log(isoDate);
		//console.log(download);
		//console.log(origRequest);
		
		var tempFname = strDefaultFname;
		var tempCNetFname = strDefaultCnetFname;
		
		switch (useDefaultFname) {
			case true:
				console.log('Using Default Filename');
				break;
			case false:
				console.log('Using Prompt as Filename');
				var strTmpPrompt = origRequest.prompt;
				if (strTmpPrompt == null) {
					console.log('Prompt is empty, reverting to default filename');
					tempFname = strDefaultFname;
					tempCNetFname = strDefaultCnetFname;
					break;
				}
				
				if (strTmpPrompt.length == 0 ) {
					console.log('Prompt is zero length, reverting to default filename');
					tempFname = strDefaultFname;
					tempCNetFname = strDefaultCnetFname;
					break;
				}
				//console.log('Old: ' + strTmpPrompt);
				var newTmpVar = strTmpPrompt.replaceAll(/[^a-zA-Z]/g, "");
				newTmpVar = newTmpVar.substring(0, maxFnameChar);
				strTmpPrompt = newTmpVar;			
				//console.log('New: ' + strTmpPrompt);
				if (strTmpPrompt == null) {
					console.log('Prompt is empty, reverting to default filename');
					tempFname = strDefaultFname;
					tempCNetFname = strDefaultCnetFname;
					break;
				}
				if (strTmpPrompt.length == 0 ) {
					console.log('Prompt is zero length, reverting to default filename');
					tempFname = strDefaultFname;
					tempCNetFname = strDefaultCnetFname;
					break;
				}
				tempFname = strTmpPrompt + "-ED-";
				tempCNetFname = strTmpPrompt + "-CNet-";
				break;
			default:
				break;
		}
		
		
		switch (ftype) {
			case "jpeg":
				fname=tempFname + download.substr(0,15) + "." + ftype;
				txtname=tempFname + download.substr(0,15) + ".json";
				break;
			case "png":
				fname=tempFname + download.substr(0,15) + "." + ftype;
				txtname=tempFname + download.substr(0,15) + ".json";
				break;
			case "gif":
				fname=tempFname + download.substr(0,15) + "." + ftype;
				txtname=tempFname + download.substr(0,15) + ".json";
				break;
			default:
				alert("Sorry the file type " + ftype + " is not supported.");
				break;
		}
		switch (fname) {
			case null:
				break;
			default:
				//console.log(fname);
				var cnetImageName = tempCNetFname + download.substr(0,15);
				const img = image;
				const canvas = document.createElement("canvas");
				const context = canvas.getContext("2d");
				const ch = img.naturalHeight;
				const cw = img.naturalWidth;
				canvas.width = cw;
				canvas.height = ch;
				context.save();
				context.drawImage(img,0,0);
				context.restore();
				//img.src = canvas.toDataURL(fname);
				let canvasImage = canvas.toDataURL('image/' +ftype);
				//console.log(canvasImage);
				let xhr = new XMLHttpRequest();
				xhr.responseType = 'blob';
				xhr.onload = function () {
					let a = document.createElement('a');
					a.href = window.URL.createObjectURL(xhr.response);
					a.download = fname;
					a.style.display = 'none';
					document.body.appendChild(a);
					a.click();
					a.remove();
				}
				xhr.open('GET', canvasImage); // This is to download the canvas Image
				xhr.send();
				
				//console.log(origRequest);
				var tmpLoraModel = origRequest.use_lora_model;
				var tmpLoraModelSwitch = [];
				var tmpLoraAlpha = origRequest.lora_alpha;
				var tmpLoraAlpahSwitch = [];
				
				switch (typeof tmpLoraModel) {
					case "string" :
						console.log('Its a string');
						tmpLoraModelSwitch[0] = tmpLoraModel;
						tmpLoraAlpahSwitch[0] = tmpLoraAlpha;
						origRequest.use_lora_model = tmpLoraModelSwitch;
						origRequest.lora_alpha = tmpLoraAlpahSwitch;
						console.log(origRequest);
						break;
					default:
						break;
				}
						

				
				let blob = new Blob([JSON.stringify(origRequest,null,2)], { type: 'application/json' });
				//console.log(blob);
				saveAs(blob, txtname);
				
				var cnetImage = origRequest.control_image;
				switch (cnetImage) {
					case undefined:
						break;
					default:
						//console.log(cnetImage);
						const cnetcollection = document.getElementsByClassName("task-fs-initimage");
						const cnetimg = (cnetcollection[0].innerHTML);
						var tempElement = document.createElement("div");
						tempElement.innerHTML = cnetimg;
						var images = tempElement.querySelectorAll("img");
						var cnetsrc=null;
						images.forEach(function (dummyimage) {
							cnetsrc = dummyimage.getAttribute("src");
						});
						tempElement.remove();
						let cnetcanvasImage = cnetsrc;
						let cnetxhr = new XMLHttpRequest();
						cnetxhr.responseType = 'blob';
						cnetxhr.onload = function () {
							let a = document.createElement('a');
							a.href = window.URL.createObjectURL(cnetxhr.response);
							a.download = cnetImageName;
							a.style.display = 'none';
							document.body.appendChild(a);
							a.click();
							a.remove();
						}
						cnetxhr.open('GET', cnetcanvasImage); // This is to download the canvas Image
						//console.log('Writing image ' + cnetImageName);
						cnetxhr.send();
						break;
				}
				break;
		}
	}

	function onRotate90CW(origRequest, image) {
		var RotateProceedFlag = null;
		switch (origRequest.showInfoBar) {
			case undefined:
				RotateProceedFlag = true;
				break;
			case true:
				RotateProceedFlag = false;
				break;
			default:
				RotateProceedFlag = true;
				break;
		}
		
			//case undefined:
			//	rotate(image,90);
			//	break;
			//case true:
			//	alert('Please turn off information before you rotate the image.');
			//	break;
			//case false:
			//	rotate(image,90);
			//	break;
			//default:
			//	rotate(image,90);
			//	break;
				
		switch (RotateProceedFlag) {
			case false:
				alert('Please turn off information before you rotate the image.');
				break;
			case true:
				rotate(image,90);
				modifyTileSetting(origRequest);
				break;
			default:
				rotate(image,90);
				modifyTileSetting(origRequest);
				break;
		}
	}


	function onRotate90CCW(origRequest, image) {
		var RotateProceedFlag = null;
		switch (origRequest.showInfoBar) {
			case undefined:
				RotateProceedFlag = true;
				break;
			case true:
				RotateProceedFlag = false;
				break;
			default:
				RotateProceedFlag = true;
				break;
		}
		switch (RotateProceedFlag) {
			case false:
				alert('Please turn off information before you rotate the image.');
				break;
			case true:
				rotate(image,-90);
				modifyTileSetting(origRequest);
				break;
			default:
				rotate(image,-90);
				modifyTileSetting(origRequest);
				break;
		}
			
	}
	
	
	function modifyTileSetting(origRequest) {
		switch (origRequest.tiling) {
			case "none":
				break;
			case "x":
				origRequest.tiling = "y";
				break;
			case "y":
				origRequest.tiling = "x";
				break;
			case "xy":
				break;
			default:
				break;
		}
		
	}

	function onFlipHoriz(origRequest, image) {
		mirror(origRequest,image,0,0,true,false);
		//console.log('*** Image Flipped Horizontally***');
	}

	function onFlipVert(origRequest, image) {
		mirror(origRequest,image,0,0,false,true);
		//console.log('*** Image Flipped Vertically***');
	}


	function onImageAdjustClick(origRequest, image) {
		console.log("**Image Adjust Clicked**");
	}
		
	function rotate(image,angle) {
		if(image.complete){ // ensure the image has loaded
			const img = image;
			const canvas = document.createElement("canvas");
			const context = canvas.getContext("2d");
  
			// Make canvas the same size as img inverted
			var cw = img.naturalHeight;
			var ch = img.naturalWidth;

			canvas.width = cw;
			canvas.height = ch;
			
			// Added Line below 
			context.save();
  
			// Draw rotated img onto canvas
			context.translate(cw / 2, ch / 2);
			context.rotate(angle * Math.PI / 180);
			context.drawImage(img, -ch / 2, -cw / 2, ch, cw);
			
			//Added Line below
			context.restore();
  
			// Replace img with canvas contents
			img.src = canvas.toDataURL();
		}
		//throw new ReferenceError("Image is not complete.");
	}

	function mirror(image, x = 0, y = 0, horizontal = false, vertical = false) {
		if(image.complete){ // ensure the image has loaded
			const img = image;
			const canvas = document.createElement("canvas");
			const context = canvas.getContext("2d");
  
			// Make canvas the same size as img
			var cw = img.naturalWidth;
			var ch = img.naturalHeight;
			
			canvas.width = cw;
			canvas.height = ch;
			
			// Added Line below
			//context.save();
  
			// Draw rotated img onto canvas
			context.setTransform(
				horizontal ? -1 : 1, 0, // set the direction of x axis
				0, vertical ? -1 : 1,   // set the direction of y axis
				x + (horizontal ? image.width : 0), // set the x origin
				y + (vertical ? image.height : 0)   // set the y origin
			);
			context.drawImage(image,0,0);
			
			// Added Line below
			context.setTransform(1,0,0,1,0,0);
			//context.restore();
  
			// Replace img with canvas contents
			img.src = canvas.toDataURL();
		}
		//throw new ReferenceError("Image is not complete.");
	}

	function makeImageDrawable(image){
		if(image.complete){ // ensure the image has loaded
			var dImage = document.createElement("canvas"); // create a drawable image
			dImage.width = image.naturalWidth;      // set the resolution
			dImage.height = image.naturalHeight;
			dImage.style.width = image.style.width; // set the display size
			dImage.style.height = image.style.height; 
			dImage.ctx = dImage.getContext("2d");   // get drawing API
                                                // and add to image
                                                // for possible later use
			dImage.ctx.drawImage(image,0,0);
			return dImage;
		}
		throw new ReferenceError("Image is not complete.");
	}
	
	
	function onDisplayInfo(origRequest, image) {
		// Get variables from prompt
		var imgInfSteps = origRequest.num_inference_steps;
		var imgGuideStr = origRequest.guidance_scale;
		var imgSeed = origRequest.seed;
		var imgSampler = origRequest.sampler_name;
		var imgRandomSeed = 'No';
		var imgClipSkip = 'No';
		var imgTiled = 'No';
		var imgModel = origRequest.use_stable_diffusion_model;
		var imgLoraModel = 'Undefined';
		var imgLoraCount = 0;
		var imgFirstLoraModel = 'Undefined';
		var imgFirstLoraValue='Nil';
		var imgVAE = origRequest.use_vae_model;
		var imgTempVal = 'Undefined';
		var subDirPos = null;
		var checkModelName = null;
		var checkLoRaName = null;


		
		//console.log(origRequest);
		
		checkModelName = imgModel.includes("/");
		switch (checkModelName) {
			case true:	
				//console.log('Found sub Directory');
				subDirPos = imgModel.lastIndexOf('/');
				imgModel= imgModel.substr(subDirPos+1);
				break;
			default:
				//console.log('No sub Directory found')
				break;
		}

		
		switch (origRequest.used_random_seed) {
			case true:
				imgRandomSeed = 'Yes';
				break;
			case false:
				imgRandomSeed = 'No';
				break;
			default:
				imgRandomSeed = 'Yes';
				break;
		}
			
		switch (origRequest.clip_skip) {
			case false:
				imgClipSkip='No';
				break;
			case true:
				imgClipSkip='Yes';
				break;
			default:
				imgClipSkip='No';
				break;
		}
		
		switch (origRequest.tiling) {
			case "none":
				imgTiled = 'No';
				break;
			case "x":
				imgTiled = 'Horiz';
				break;
			case "y":
				imgTiled = 'Vert';
				break;
			case "xy":
				imgTiled = 'Both';
				break;
			default:
				imgTiled = 'No';
				break;
		}
		
		var imgLora = origRequest.lora_alpha;
		//console.log('Lora_Alpha: ' + imgLora);
		if (typeof imgLora == 'undefined') {
			imgLora = 'undefined';
		}
		switch (imgLora) {
			case undefined:
				imgLoraModel = 'Not Used';
				imgLora = 'N/A';
				break;
			case 'undefined':
				//console.log('LoRa was undefined');
				imgLoraModel = 'Not Used';
				imgLora = 'N/A';
				break;
			case '0':
				imgLora=0;
				break;
			default:
				imgLoraModel = origRequest.use_lora_model;
				imgLora=origRequest.lora_alpha;
				break;
		}
		
		if (typeof imgLoraModel == 'object') {
			imgFirstLoraModel = imgLoraModel[0];
			imgFirstLoraValue = imgLora[0];
			imgTempVal = imgLoraModel[0].length;
			imgLoraCount = imgLoraModel.length;
		} else {
			imgFirstLoraModel = imgLoraModel;
			imgFirstLoraValue = imgLora;
			imgTempVal = imgLoraModel.length;
			imgLoraCount = 1;
		}
		
		if (imgTempVal > 15) {
			imgFirstLoraModel = imgFirstLoraModel.slice(0,15) + '....';
		}
			
		//console.log('Lora: ' + imgFirstLoraModel);
		//console.log('Length: ' + imgTempVal);
		
		//console.log('Number of Lora: ' + imgLoraCount);

		var imgHyper = origRequest.hypernetwork_strength;
		//console.log(origRequest.hypernetwork_strength);
		switch (imgHyper) {
			case 'undefined':
				imgHyper = 'No';
				break;
			case '0':
				imgHyper=0;
				break;
			default:
				imgHyper=origRequest.hypernetwork_strength;
				break;
		}
		
		var imgCNet = origRequest.control_filter_to_apply;
		//console.log('CNet: ' + origRequest.control_filter_to_apply);
		switch (imgCNet) {
			case undefined:
				imgCNet = 'Not Used';
				break;
			case 'undefined':
				imgCNet = 'No';
				break;
			default:
				imgCNet = origRequest.control_filter_to_apply;
				break;
		}
			
			
			


		
		var LoraAlphaValue = imgLora; 		//origRequest.lora_alpha;
		//console.log('Hype: ' + imgHyper);
		var checkWidth = image.naturalWidth;
		var proceedFlag=false;
		var toggleInfo=null;
		var toggleSamplerBar=null;
		var cw = 0;
		var ch = 0;
		var bw = 0;
		var bh1 = 0;
		var bh2 = 0;
		
		toggleInfo=origRequest.showInfoBar;
		switch (toggleInfo) {
			case undefined:
				toggleInfo=true;
				break;
			case true:
				toggleInfo=false;
				break;
			case false:
				toggleInfo=true;
				break;
			default:
				toggleInfo=false;
				break;
		}
		origRequest.showInfoBar = toggleInfo;
			
		if (typeof checkWidth === 'string') {
			checkWidth=parse(checkWidth);
		}
		if (checkWidth >448) {
			proceedFlag=true;
		}
		
		switch (proceedFlag) {
			case false:
				alert('Image is too small to display information');
				break;
			default:
				//Create the canvas to work with
				const img = image;
				const canvas = document.createElement("canvas");
				const context = canvas.getContext("2d");
				cw = img.naturalWidth;
				bw = cw;
		
				// Make the canvas slightly larger than the original image
				// to accomodate the information bar
				//const cw = img.naturalWidth;
				switch (toggleInfo) {
					case true:
						ch = img.naturalHeight+80;
						ch = ch + (imgLoraCount*20);
						bh1 = img.naturalHeight;
						bh2 = ch + 80;
						break;
					case false:
						ch = img.naturalHeight-80;
						ch = ch - (imgLoraCount*20);
						break;
				}
				
				canvas.width = cw;
				canvas.height = ch;
				
				
			
				// Display the canvas	
				context.save();
				context.drawImage(img,0,0);
				
				switch (toggleInfo) {
					case true:
						// Add rectangle to base of canvas
						context.beginPath();
						//context.rect(0,ch-80, cw,80);
						context.rect(0,bh1,cw,bh2);
						context.fillStyle = 'black';
						context.fill();
						context.lineWidth = 7;
						context.strokeStyle = 'black';
						context.stroke();
						if (imgModel.length > 30) {
							imgModel = imgModel.slice(0,30) + '...';
						}
						
						if (imgSampler.length > 15) {
							imgSampler = imgSampler.slice(0,15) + '...';
						}
						
		
						// Add text info to the rectangle
						context.fillStyle = 'white';
						context.font = '16px serif';
						context.fillText('Model: ' + imgModel, 20, bh1+20);
						context.fillText('Sampler: ' + imgSampler, 320, bh1+20);
						context.fillText('IS: ' + imgInfSteps, 20, bh1+40);
						context.fillText('GS: ' + imgGuideStr, 80, bh1+40);
						context.fillText('Clip Skip: ' + imgClipSkip, 20, bh1+60);
						context.fillText('Tile: ' + imgTiled, 160, bh1+60);
						context.fillText('CNet: ' + imgCNet, 320, bh1+60);
						context.fillText('Seed: ' + imgSeed, 320,bh1+40);
						//context.fillText('Hype: ' + imgHyper, 20, bh1+60);
						//context.fillText('Lora: ' + imgFirstLoraModel, 160, bh1+60);
						//context.fillText('Value: ' + imgFirstLoraValue, 320, bh1+60);
						context.fillText('Random Seed: ' + imgRandomSeed,160, bh1+40);
						
						if (imgLoraCount == 1) {
							var tempCount = 1;
							var tempStr=null;
							var xpos = 20;
							var ypos = ch - 20;
							tempStr = imgLoraModel;
							var tempWeight = imgLora;
							if (tempStr.length > 25) {
								tempStr=tempStr.slice(0,25) + '....';
							}
							context.fillText('Lora #' + tempCount + ':', xpos, ypos);
							context.fillText(tempStr, xpos+60, ypos);
							xpos = 320;
							context.fillText('Value: ' + tempWeight, xpos, ypos);
						}
							
							
						
						if (imgLoraCount >1) {
							var tempCount = 1;
							var tempStr=null;
							var xpos = 20;
							var ypos = ch - 20;
							for (const [key, value] of Object.entries(imgLoraModel)) {
								tempStr = imgLoraModel[(imgLoraCount - tempCount)];
								checkLoRaName = tempStr.includes("/");
								switch (checkLoRaName) {
									case true:	
										//console.log('LoRa Found sub Directory');
										subDirPos = tempStr.lastIndexOf('/');
										tempStr= tempStr.substr(subDirPos+1);
										break;
									default:
										//console.log('No sub Directory found')
										break;
								}
								
								
								
								
								
								if (tempStr.length > 25) {
									tempStr=tempStr.slice(0,25) + '....';
								}
								context.fillText('Lora #' + ((imgLoraCount - tempCount)+1) + ':', xpos, ypos);
								context.fillText(tempStr, xpos+60, ypos);
								//context.fillText(imgLoraModel[(imgLoraCount - tempCount)], xpos+60, ypos);
								ypos = ypos - 20;
								tempCount ++;
								//console.log(key, value);  // first one, second two
							}
							tempCount = 1;
							tempStr=null;
							xpos = 320;
							ypos = ch - 20;
							for (const [key, value] of Object.entries(imgLora)) {
								tempStr = imgLora[(imgLoraCount - tempCount)];
								context.fillText('Value: ' + tempStr, xpos, ypos);
								ypos = ypos - 20;
								tempCount ++
							}
								
						}
						context.restore();
						break;
					case false:
						break;
				}
				// Replace img with canvas contents
				img.src = canvas.toDataURL();
				break;
		}
	}

	
	function onTileImage(origRequest,image) {
		var TileProceedFlag = null;
		switch (origRequest.showInfoBar) {
			case undefined:
				TileProceedFlag = true;
				break;
			case true:
				TileProceedFlag = false;
				break;
			default:
				TileProceedFlag = true;
				break;
		}
				
		switch (TileProceedFlag) {
			case false:
				alert('Please turn off information before you tile the image.');
				break;
			case true:
				onTileTheImage(origRequest,image);
				break;
			default:
				onTileTheImage(origRequest,image);
				break;
		}
			
			

		
		//switch (origRequest.showInfoBar) {
		//	case undefined:
		//		onTileTheImage(origRequest,image);
		//		break;
		//	case true:
		//		alert('Please turn off information before you tile the image.');
		//		break;
		//	case false:
		//		onTileTheImage(origRequest,image);
		//		break;
		//	default:
		//		onTileTheImage(origRequest,image);
		//		break;
		//}
		
	}
	
	
	function onTileTheImage(origRequest,image) {
		var myTileDir = origRequest.tiling;
		const img = image;
		const canvas = document.createElement("canvas");
		const context = canvas.getContext("2d");
		var cw = img.naturalWidth;
		var ch = img.naturalHeight;
		
		switch (myTileDir) {
			case 'x':
				canvas.width = cw + cw;
				canvas.height = ch;
				context.save();
				context.drawImage(img,0,0); 			// Draw image at origin
				context.drawImage(img,cw,0);			// Draw image along side origin
				context.restore();
				img.src = canvas.toDataURL();
				break;
			case 'y':
				canvas.width = cw;
				canvas.height = ch + ch;
				context.save();
				context.drawImage(img,0,0); 			// Draw image top left
				context.drawImage(img,0,ch);			// Draw image below origin
				context.restore();
				img.src = canvas.toDataURL();
				break;
			case 'xy':
				canvas.width = cw + cw;
				canvas.height = ch + ch;
				context.save();
				context.drawImage(img,0,0); 			// Draw image at origin
				context.drawImage(img,cw,0);			// Draw image along side origin
				context.drawImage(img,0,ch);			// Draw image below origin
				context.drawImage(img,cw,ch);			// Draw image diagonal opposite origin
				context.restore();
				img.src = canvas.toDataURL();
				break;
			default:
				//console.log('No tiling selected');
				alert('Cannot tile this image - Seamless Tiling has not been set for this image.');
				break;
		}
			
	}
		
	function onSaveMask(origRequest, image) {
		
	}
	
	function onInvertMask(origRequest,image) {
		
	}
	
	


})();