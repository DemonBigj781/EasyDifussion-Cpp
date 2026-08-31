/**
 * Image to img2img
 * v.1.1, last updated: 04/12/2023
 * By The Stig
 *
 * Change Log
 * 04/12/2023 Bug Fix in Zoom trapping Zero Values
 * 04/12/2023 Bug Fix in Rotate, typo error
 * 04/12/2023 Bug fix in UI - incorrect hover over displayed for Move/Pan section
 * 01/12/2023 Added Move / Pan function
 * 29/11/2023 Added Rotate function
 * 25/11/2023 Fixed Zoom function to center image
 * 23/11/2023 Added Zoom Function (Experimental)
 * 18/11/2023 Added LoRa Adjustment routine
 * 18/11/2023 Removed Seed toggle option
 * 17/11/2023 Experimental img2img routine added
 * 17/11/2023 Added checkReadyStatus function
 * 14/11/2023 Bug fix - stopBatch wasn't triggering Reset Variables function
 * 13/11/2023 Added Start and Stop buttons to head of DOM as requested by SmartDaniel
 * 13/11/2023 Bug fix - incorrect settings for Toggle buttons
 * 11/11/2023 Added button to set all adjustment boxes to Zero.
 * 11/11/2023 Added Option to use Sequential Prompts from the Wildcard plugin.
 * 11/11/2023 Added Wait Timer to wait for a task to be completed before starting the next task
 * 10/11/2023 Added integration to Build A Scene plugin
 * 10/11/2023 Added integration to Dress Me Up plugin
 * 10/11/2023 Added integration to Wildcard Text to Prompt plugin
 * 09/11/2023 Added Timeout setting - thanks to SmartDaniel for assistance
 * 09/11/2023 Added 'Still to process' text box
 * 08/11/2023 Added routine to cancel all tasks
 * 08/11/2023 Beta Fixed the resolution problem with img2img preview
 * 07/11/2023 Added routine to enable Seed Adjustment
 * 07/11/2023 Fixed Initial Guidance and Prompt bug
 * 05/11/2023 Added routine to trap Guide Scale and Prompt Strength do not exceed minimum and maximum values
 * 05/11/2023 First Beta release to public
 * 01/11/2023 Initial Build
 *
 * Free to use with the CMDR2 Stable Diffusion UI.
 *
 */
(function() { "use strict"
    const VERSION = "1.1";
    const ID_PREFIX = "TheStig-BuildAScene-plugin";
    console.log('%s Embed Metadata Version: %s', ID_PREFIX, VERSION);


	// Change the following variables for the Default Settings
	var switchStatus = true;
	var adjustSeedStatus = true;
	var BuildSceneStatus = false;
	var DressMeUpStatus = false;
	var Text2PromptStatus = false;
	var Text2PromptSeqStatus = false;

	var BuildSceneEnabled = false;
	var DressMeUpEnabled = false;
	var Text2PromptSeqEnabled = false;

	var firstWildcardPass = true;

	var defaultBatchCount = 4;

	var defaultGuideScaleAdjust = 2;
	var defaultGuideScaleStep =  0.2;
	var defaultGuideScalePreset =  0;

	var defaultPromptStrengthAdjust = 1;
	var defaultPromptStrengthStep =  0.06;
	var defaultPromptStrengthReset =  8;

	var defaultLoRaStrength = 1;
	var defaultLoRaStrengthAdjust = 0.05;
	var defaultLoRaStrengthPreset = 0;

	var defaultPromptChange = 8;


	var defaultTimeOut = 1000;



	var minGuide = 1.1;
	var maxGuide = 50;

	var minPrompt = 0;
	var maxPrompt = 0.99;

	var minLoRa = -50;
	var maxLoRa = 50;

	var minSeed = 0;

	var stopBatch = false;

	var currGuide = Number(document.getElementById ("guidance_scale").value);
	var oldGuide = Number(document.getElementById ("guidance_scale").value);

	var currPrompt = Number(document.getElementById ("prompt_strength").value);
	var oldPrompt = Number(document.getElementById ("prompt_strength").value);

	var currLoRaModel = document.getElementById("lora_0").value;
	var currLoRaStrength = Number(document.querySelector(".model_weight").value);
	var oldLoRaStrength = Number(document.querySelector(".model_weight").value);

	//var oldLoRa = Number();
	//var LoraModel=document.getElementById("lora_0").value;
	//var tempModel = document.getElementsByClassName("model_entry");
	//tempModel.item(LoraSelected-1).querySelector(".model_weight").value = parseFloat(newSetVal);


	var currSeed = Number(document.getElementById ("seed").value);
	var oldSeed = Number(document.getElementById ("seed").value);



	var image2imgTaskCount = 0;

	injectLoaderCSS();
	addBuildOptions();
	addSecondaryBuildOptions();

	var myBatchCount = Number(document.getElementById ("image2imgBatchCount").value);
	var myGuideStr = Number(document.getElementById ("image2imgAdjGuideStr").value);
	var myGuideAdj = Number(document.getElementById ("image2imgAdjGuideVal").value);
	var myPromptStr = Number(document.getElementById ("image2imgAdjPromptStr").value);
	var myPromptReset = Number(document.getElementById ("image2imgResetPrompt").value);
	var myPromptAdj = Number(document.getElementById ("image2imgAdjPromptVal").value);
	var myLoRaStr = Number(document.getElementById ("image2imgAdjLoRa").value);
	var myLoRaAdj = Number(document.getElementById ("image2imgAdjLoRaVal").value);
	var mySeedStr = Number(document.getElementById ("image2imgAdjSeed").value);
	var mySeedAdj = Number(document.getElementById ("image2imgAdjSeedVal").value);
	var myWarpCount = Number(document.getElementById ("image2imgWarpEvery").value);
	var myRotateCount = Number(document.getElementById ("image2imgRotateEvery").value);
	var myPanCount = Number(document.getElementById ("image2imgPanEvery").value);
	var myPromptChange = Number(document.getElementById ("image2imgChangePrompt").value);

	var zoomFlag = false;
	var rotateFlag = false;
	var panFlag = false;
	var transitionOptions = "";


	var guideCount = 1;
	var promptCount = 1;
	var promptReset = 0;
	var loraCount = 1;
	var seedCount = 1;
	var promptChangeCount = 1;
	var warpCount = 0;
	var rotateCount = 0;
	var panCount=0;


	function injectLoaderCSS() {
		console.log('Initialising img2img CSS');
		const style = document.createElement('style');
		style.textContent = `
			.image2imgStart {
				background-color:#2168bf;
			}

			.image2imgStop {
				background-color:#840800;
			}

			.image2imgVars {
				background-color:#123456;
				display: none;
			}

			.inputWidth {
				width:280px;
				min-width: 280px;
				max-width: 280px ;
				overflow: hidden;
			}

			.imglabelWidth1 {
				width:160px  !important;
				display: inline-block;
				overflow: hidden;
			}

			.imglabelWidth2 {
				width:200px  !important;
				display: inline-block;
				overflow: hidden;
			}

			.materialWidth {
				width:120px;
				min-width: 120px;
				max-width: 120px ;
				overflow: hidden;
			}

			.colorWidth {
				width:120px;
				min-width: 120px;
				max-width: 120px ;
				overflow: hidden;
			}

			.txtBox {
				resize: none;
			}

			.txtBox2 {
				resize: both;
				 overflow: auto;
			}

			#confirm {
				display: none;
				background-color: white;
				font-size: 20px;
				color: blue;
				border: 4px solid;
				border-radius: 12px;
				padding: 5px;
				position: fixed;
				height: 120px;
				width: 240px;
				left: 50%;
				top: 50%;
				padding: 6px 8px 8px;
				text-align: center;
			}

			.close {
				display: flex;
				position: absolute;
				right: 20px;
				bottom: 8px;
			}

			#confirmRemove {
				display: none;
				background-color: white;
				font-size: 20px;
				color: blue;
				border: 4px solid;
				border-radius: 12px;
				padding: 5px;
				position: fixed;
				height: 120px;
				width: 240px;
				left: 50%;
				top: 50%;
				padding: 6px 8px 8px;
				text-align: center;
			}

			.removeItems {
				display: flex;
				position: absolute;
				right: 20px;
				bottom: 8px;
			}

			.slidecontainer {
				width: 100%; /* Width of the outside container */
			}

			.slider {
				-webkit-appearance: none;  /* Override default CSS styles */
				appearance: none;
				width: 80%; /* Full-width */
				height: 4px; /* Specified height */
				background: #b0b0ff; /* Light Blue background */
				outline: none; /* Remove outline */
				opacity: 0.7; /* Set transparency (for mouse-over effects on hover) */
				-webkit-transition: .2s; /* 0.2 seconds transition on hover */
				transition: opacity .2s;
				display: none;
			}


			.slider:hover {
				opacity: 1; /* Fully shown on mouse-over */
			}


			.slider::-webkit-slider-thumb {
				-webkit-appearance: none; /* Override default look */
				appearance: none;
				width: 16px; /* Set a specific slider handle width */
				height: 16px; /* Slider handle height */
				background: #04AA6D; /* Green background */
				cursor: pointer; /* Cursor on hover */
			}

			.slider::-moz-range-thumb {
				width: 16px; /* Set a specific slider handle width */
				height: 16px; /* Slider handle height */
				background: #04AA6D; /* Green background */
				cursor: pointer; /* Cursor on hover */
			}

			/* The switch - the box around the slider */
			.switch {
				position: relative;
				display: inline-block;
				width: 60px;
				height: 34px;
			}

			/* Hide default HTML checkbox */
			.switch input {
				opacity: 0;
				width: 0;
				height: 0;
			}

			/* The slider */
			.sliderToggle {
				position: absolute;
				cursor: pointer;
				top: 0;
				left: 0;
				right: 0;
				bottom: 0;
				background-color: #ccc;
				-webkit-transition: .4s;
				transition: .4s;
			}

			.sliderToggle:before {
				position: absolute;
				content: "";
				height: 26px;
				width: 26px;
				left: 4px;
				bottom: 4px;
				background-color: white;
				-webkit-transition: .4s;
				transition: .4s;
			}

			input:checked + .sliderToggle {
				background-color: #2196F3;
			}

			input:focus + .sliderToggle {
				box-shadow: 0 0 1px #2196F3;
			}

			input:checked + .sliderToggle:before {
				-webkit-transform: translateX(26px);
				-ms-transform: translateX(26px);
				transform: translateX(26px);
			}

			/* Rounded sliders */
			.sliderToggle.round {
				border-radius: 34px;
			}

			.sliderToggle.round:before {
				border-radius: 50%;
			}

		`;
		document.head.appendChild(style);
	}

	function addBuildOptions() {
		console.log('Initialising img2img Settings');
		var image2imgOpenCheck2 = '';
		image2imgOpenCheck2 = ' active';
		var BuildSettings = document.createElement('div');
		BuildSettings.id = 'image2img-settings';
		BuildSettings.classList.add('panel-box');
		let tempHTML = `
			<h4 class="collapsible `+ image2imgOpenCheck2 +`">Image to img2img</h4>
			<div id="image2img-settings-entries" class="collapsible-content" style="display: block;margin-top:15px;">
				<label for="image2imgBatchCount" class="imglabelWidth2">Number of generations:</label>
				<input type="number" title="Number of images to create" class="imglabelWidth1" id="image2imgBatchCount" value="4" min="1">
				<p></p>
				<label for="image2imgGSVal" class="imglabelWidth2">Guidance Scale (ref only)</label>
				<input type="number" title="Reference only" class="imglabelWidth1" id="image2imgGSVal" value="0.7" min="1.1" max="50">
				<br>
				<div class="slidecontainer">
				<input type="range" title="Reference only" min="11" max="500" value="250" class="slider" id="image2imgGuideSlider">
				</div>
				<label for="image2imgAdjGuideStr" class="imglabelWidth2">Adjust Guide Scale every</label>
				<input type="number" title="Image count to adjust the Guide Scale" class="imglabelWidth1" id="image2imgAdjGuideStr" value="2" min="0">

				<br>
				<label for="image2imgAdjGuideVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustment value for Guide Scale" class="imglabelWidth1" id="image2imgAdjGuideVal" value="0.2">

				<label for="image2imgResetGuide" class="imglabelWidth2">Reset Guide Scale every</label>
				<input type="number" title="Reset Guide Scale every" class="imglabelWidth1" id="image2imgResetGuide" value="0">

				<p></p>
				<label for="image2imgPSVal" class="imglabelWidth2">Prompt Strength (ref only)</label>
				<input type="number" title="Reference only" class="imglabelWidth1" id="image2imgPSVal" value="2" min="0" max="10">

				<div class="slidecontainer">
				<input type="range" title="Reference only" min="0" max="100" value="50" class="slider" id="image2imgPromptSlider">
				</div>
				<label for="image2imgAdjPromptStr" class="imglabelWidth2">Adjust Prompt Strength every</label>
				<input type="number" title="Image count to adjust the Prompt Strength" class="imglabelWidth1" id="image2imgAdjPromptStr" value="2" min="0">
				<br>
				<label for="image2imgAdjPromptVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustment value for Prompt Strength" class="imglabelWidth1" id="image2imgAdjPromptVal" value="0.02">

				<label for="image2imgResetPrompt" class="imglabelWidth2">Reset Prompt Strength every</label>
				<input type="number" title="Reset Prompt Strength every" class="imglabelWidth1" id="image2imgResetPrompt" value="0">

				<p></p>
				<label for="image2imgLoRaVal" class="imglabelWidth2">LoRa Strength (ref only)</label>
				<input type="number" title="Reference only" class="imglabelWidth1" id="image2imgLoRaVal" value="2" min="0" max="10">

				<label for="image2imgAdjLoRa" class="imglabelWidth2">Adjust LoRa Strength every</label>
				<input type="number" title="Image count to adjust the LoRa Strength" class="imglabelWidth1" id="image2imgAdjLoRa" value="1" min="0">
				<br>
				<label for="image2imgAdjLoRaVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustment value for LoRa Strength" class="imglabelWidth1" id="image2imgAdjLoRaVal" value="0.02">

				<label for="image2imgResetLora" class="imglabelWidth2">Reset LoRa Strength every</label>
				<input type="number" title="Reset LoRa Strength every" class="imglabelWidth1" id="image2imgResetLora" value="0">
				<p></p>
				<!--
				<div class="sliderwrapper">
				<label style="display:inline" title="Set Adjust Seed On/Off" class="switch">Adjust Seed</label>
				<label class="switch">
				<input type="checkbox" id="image2imgSeed">
				<span class="slider round"></span>
				</label>
				</div>
				-->
				<label for="image2imgAdjSeed" class="imglabelWidth2">Adjust Seed value every</label>
				<input type="number" title="Image count to adjust the Seed Value" class="imglabelWidth1" id="image2imgAdjSeed" value="2" min="0">
				<br>
				<label for="image2imgAdjSeedVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustment value for Seed" class="imglabelWidth1" id="image2imgAdjSeedVal" value="1">

				<label for="image2imgResetSeed" class="imglabelWidth2">Reset Seed every</label>
				<input type="number" title="Reset Seed every" class="imglabelWidth1" id="image2imgResetSeed" value="0">
				<p></p>

				<label for="image2imgWarpEvery" class="imglabelWidth2">Zoom image every</label>
				<input type="number" title="Image count to adjust the Zoom" class="imglabelWidth1" id="image2imgWarpEvery" value="1">

				<label for="image2imgResetWarp" class="imglabelWidth2">Reset Zoom Factor every</label>
				<input type="number" title="Reset Zoom Factor every" class="imglabelWidth1" id="image2imgResetWarp" value="0">

				<p></p>
				<label for="image2imgWarpWidth" class="imglabelWidth2">Zoom Width Factor</label>
				<input type="number" title="Zoom Warp Factor" class="imglabelWidth1" id="image2imgWarpWidth" min="0.001" value="1.05">

				<label for="image2imgWarpWidthAdjVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustmentment value for Width" class="imglabelWidth1" id="image2imgWarpWidthAdjVal" value="0">

				<label for="image2imgWarpWidthAdj" class="imglabelWidth2">Adjust Zoom Width every</label>
				<input type="number" title="Adjust the Zoom Width every" class="imglabelWidth1" id="image2imgWarpWidthAdj" value="0">
				<p></p>
				<label for="image2imgWarpHeight" class="imglabelWidth2">Zoom Height Factor</label>
				<input type="number" title="Height Zoom Factor" class="imglabelWidth1" id="image2imgWarpHeight" min="0.001" value="1.05">

				<label for="image2imgWarpHeightAdjVal" class="imglabelWidth2">Adjustment Value</label>
				<input type="number" title="Adjustmentment value for Height" class="imglabelWidth1" id="image2imgWarpHeightAdjVal" value="0">

				<label for="image2imgWarpHeightAdj" class="imglabelWidth2">Adjust Zoom Height every</label>
				<input type="number" title="Adjust the Zoom Height every" class="imglabelWidth1" id="image2imgWarpHeightAdj" value="0">

				<p></p>

				<label for="image2imgRotateEvery" class="imglabelWidth2">Rotate image every</label>
				<input type="number" title="Rotate every" class="imglabelWidth1" id="image2imgRotateEvery" value="0">

				<label for="image2imgRotateAngle" class="imglabelWidth2">Rotation Angle</label>
				<input type="number" title="Rotation Angle" class="imglabelWidth1" id="image2imgRotateAngle" value="0">
				<p></p>

				<label for="image2imgPanEvery" class="imglabelWidth2">Move/Pan image every</label>
				<input type="number" title="Move/Pan every" class="imglabelWidth1" id="image2imgPanEvery" value="0">

				<label for="image2imgPanX" class="imglabelWidth2">Move/Pan X Axis</label>
				<input type="number" title="Move/Pan X-Axis" class="imglabelWidth1" id="image2imgPanX" value="0">

				<label for="image2imgPanY" class="imglabelWidth2">Move/Pan Y Axis</label>
				<input type="number" title="Move/Pan Y-Axis" class="imglabelWidth1" id="image2imgPanY" value="0">

				<p></p>
				<button type="button" class ="image2imgStop" title="Set all Adjustments to Zero" id="image2imgZeroVals">Set all Adjustment boxes to Zero</button>
				<p></p>
				<div class="sliderwrapper">
					<label style="display:inline" title="Set img2img On/Off" class="switch">Use as img2img source</label>
					<label class="switch">
						<input type="checkbox" id="image2imgSource">
						<span class="slider round"></span>
						</label>
				</div>
				<div class="sliderwrapper">
					<label style="display:inline" title="Set link to Build A Scene plugin On/Off" class="switch">Random prompt from Build A Scene plugin</label>
					<label class="switch">
					<input type="checkbox" id="image2imgBuildScene">
					<span class="slider round"></span>
					</label>
				</div>
				<div class="sliderwrapper">
					<label style="display:inline" title="Set link to Dress Me Up plugin On/Off" class="switch">Random prompt from Dress Me Up plugin</label>
					<label class="switch">
					<input type="checkbox" id="image2imgDressMeUp">
					<span class="slider round"></span>
					</label>
				</div>
				<div class="sliderwrapper">
					<label style="display:inline" title="Set link to Wildcard (Random Prompt) plugin On/Off" class="switch">Random prompt from Wildcard plugin</label>
					<label class="switch">
					<input type="checkbox" id="image2imgText2Prompt">
					<span class="slider round"></span>
					</label>
				</div>
				<div class="sliderwrapper">
					<label style="display:inline" title="Set link to Wildcard (Sequential Prompt) plugin On/Off" class="switch">Sequential prompt from Wildcard plugin</label>
					<label class="switch">
					<input type="checkbox" id="image2imgSeqText2Prompt">
					<span class="slider round"></span>
					</label>
				</div>
				<label for="image2imgChangePrompt" class="imglabelWidth2">Change prompt every</label>
				<input type="number" title="Image count to change the prompt" class="imglabelWidth1" id="image2imgChangePrompt" value="2" min="0">
				<p></p>
				<label for="image2imgTimeOut" class="imglabelWidth2">Timeout delay between tasks</label>
				<input type="number" title="Timeout delay between tasks" class="imglabelWidth1" id="image2imgTimeOut" value="100" min="100">
				<button type="button" class = "image2imgVars" id="image2imgGetVars">Get Initial Values</button>
				<p></p>
				<button type="button" class = "image2imgStart" title="Start Tasks" id="image2imgStartBatch">Start Generating Images</button>
				<p></p>
				<button type="button" class = "image2imgStop" title="Stop Tasks" id="image2imgStopBatch">Stop Generating Images</button>
				<p></p>
				<label for="image2imgOutstanding" class="imglabelWidth2">Still to process</label>
				<input type="number" title="How many images left to create" class="imglabelWidth1" id="image2imgOutstanding" value="0">
				<p></p>
				<button type="button" class = "image2imgStop" title="For Test Purposes" id="image2imgTest">Test Routine</button>
				<p></p>
			</div>
			`;
		BuildSettings.innerHTML = tempHTML;
		var editorSettings = document.getElementById('editor-settings');
		editorSettings.parentNode.insertBefore(BuildSettings, editorSettings.nextSibling);
		createCollapsibles(BuildSettings);


		document.getElementById ("image2imgStartBatch").addEventListener ("click", image2imgStartBatch, false);
		document.getElementById ("image2imgGetVars").addEventListener ("click", image2imgGetVars, false);
		document.getElementById ("image2imgStopBatch").addEventListener ("click", image2imgStopBatch, false);
		document.getElementById ("image2imgZeroVals").addEventListener ("click", image2imgZeroVals, false);

		//Setting the Default values to the UI
		document.getElementById("image2imgSource").checked = switchStatus;
		//document.getElementById("image2imgSeed").checked = adjustSeedStatus;

		document.getElementById("image2imgBatchCount").value = defaultBatchCount;

		document.getElementById("image2imgAdjGuideStr").value = defaultGuideScaleAdjust;
		document.getElementById("image2imgAdjGuideVal").value = defaultGuideScaleStep;
		document.getElementById("image2imgResetGuide").value = defaultGuideScalePreset;

		document.getElementById("image2imgAdjPromptStr").value = defaultPromptStrengthAdjust;
		document.getElementById("image2imgAdjPromptVal").value = defaultPromptStrengthStep;
		document.getElementById("image2imgResetPrompt").value = defaultPromptStrengthReset;

		document.getElementById("image2imgAdjLoRa").value = defaultLoRaStrength;
		document.getElementById("image2imgAdjLoRaVal").value = defaultLoRaStrengthAdjust;
		document.getElementById("image2imgResetLora").value = defaultLoRaStrengthPreset;

		document.getElementById("image2imgChangePrompt").value = defaultPromptChange;


		currGuide = Number(document.getElementById ("guidance_scale").value);
		oldGuide = Number(document.getElementById ("guidance_scale").value);
		document.getElementById("image2imgGSVal").value = currGuide;
		document.getElementById("image2imgGuideSlider").value = currGuide*10;

		currPrompt = Number(document.getElementById("prompt_strength").value);
		oldPrompt = Number(document.getElementById("prompt_strength").value);
		document.getElementById("image2imgPSVal").value = currPrompt;
		document.getElementById("image2imgPromptSlider").value = currPrompt*10;

		currSeed = Number(document.getElementById("seed").value);
		oldSeed = Number(document.getElementById("seed").value);

		currLoRaModel = document.getElementById("lora_0").value;
		currLoRaStrength = Number(document.querySelector(".model_weight").value);
		oldLoRaStrength = Number(document.querySelector(".model_weight").value);
		document.getElementById("image2imgLoRaVal").value = currLoRaStrength;

		document.getElementById("image2imgGSVal").disabled = true;
		document.getElementById("image2imgGuideSlider").disabled = true;
		document.getElementById("image2imgPSVal").disabled = true;
		document.getElementById("image2imgPromptSlider").disabled = true;
		document.getElementById("image2imgLoRaVal").disabled = true;
		document.getElementById("image2imgOutstanding").disabled = true;

		document.getElementById("image2imgResetGuide").disabled = true;
		//document.getElementById("image2imgResetPrompt").disabled = true;
		document.getElementById("image2imgResetLora").disabled = true;
		document.getElementById("image2imgResetSeed").disabled = true;

		document.getElementById("image2imgTimeOut").value = defaultTimeOut;


		image2imgBuildScene.addEventListener("change", imageBuildScene);
		image2imgDressMeUp.addEventListener("change", imageDressMeUp);
		image2imgText2Prompt.addEventListener("change", imageText2Prompt);
		image2imgSeqText2Prompt.addEventListener("change", imageText2PromptSeq);

		//document.getElementById("image2imgTest").addEventListener("click", image2imgTest, false);
		document.getElementById("image2imgTest").disabled = true;

		document.getElementById("image2imgResetWarp").disabled = true;

		document.getElementById("image2imgWarpWidthAdjVal").disabled = true;
		document.getElementById("image2imgWarpWidthAdj").disabled = true;

		document.getElementById("image2imgWarpHeightAdjVal").disabled = true;
		document.getElementById("image2imgWarpHeightAdj").disabled = true;

	}

	function addSecondaryBuildOptions() {
		const Image2imgEditorInputs = document.getElementById("editor-inputs");
		const Image2imgbuttonsContainer = document.createElement('div');
		Image2imgbuttonsContainer.id = `${ID_PREFIX}-Image2imgContainer`;
		Image2imgEditorInputs?.appendChild(Image2imgbuttonsContainer);

		const Image2imgStartBatch2Button = document.createElement('button');
		Image2imgStartBatch2Button.id = `Image2imgStartBatch2`;
		Image2imgStartBatch2Button.innerHTML = `Start Image2img Plugin`;
		Image2imgStartBatch2Button.title = `Start Image2img Plugin`;

		const Image2imgStopBatch2Button = document.createElement('button');
		Image2imgStopBatch2Button.id = `Image2imgStopBatch2`;
		Image2imgStopBatch2Button.innerHTML = `Stop Image2img Plugin`;
		Image2imgStopBatch2Button.title = `Stop Image2img Plugin`;

		Image2imgbuttonsContainer.appendChild(Image2imgStartBatch2Button);
		Image2imgbuttonsContainer.appendChild(Image2imgStopBatch2Button);
		document.getElementById ("Image2imgStartBatch2").addEventListener ("click", image2imgStartBatch, false);
		document.getElementById("Image2imgStartBatch2").classList = "image2imgStart";

		document.getElementById ("Image2imgStopBatch2").addEventListener ("click", image2imgStopBatch, false);
		document.getElementById("Image2imgStopBatch2").classList = "image2imgStop";

		var myTempVar = getSetting('save_to_disk');
		const image2imgtoolsMenu = document.getElementById("preview-tools");
		const image2imgSaveStatus = document.createElement('input');
		image2imgSaveStatus.setAttribute("id", "image2imgSaveStatus");
		switch (myTempVar) {
			case true:
				image2imgSaveStatus.value = "Save Status is On";
				image2imgSaveStatus.style.backgroundColor = '#13750d';
				break;
			case false:
				image2imgSaveStatus.value = "Save Status is Off";
				image2imgSaveStatus.style.backgroundColor = '#840800';
				break;
		}
		image2imgSaveStatus.style.visibility='hidden';
		image2imgSaveStatus.style.width='240px';
		image2imgSaveStatus.style.overflow='hidden';
		image2imgtoolsMenu.appendChild(image2imgSaveStatus);
	}


	function image2imgStartBatch() {
		zoomFlag = false;
		rotateFlag = false;
		stopBatch = false;
		firstWildcardPass = true;
		transitionOptions = "";
		var myTempVar = getSetting('save_to_disk');
		var myTempVar2 = document.getElementById("preview-tools");
		//console.log(myTempVar2); //.item(0).querySelector(".image2imgSaveStatus").value);
		switch (myTempVar) {
			case true:
				document.getElementById ("image2imgSaveStatus").value = "Save Status is On";
				document.getElementById("image2imgSaveStatus").style.backgroundColor = '#13750d';
				document.getElementById("image2imgSaveStatus").style.visibility='visible';
				break;
			case false:
				document.getElementById("image2imgSaveStatus").value = "Save Status is Off";
				document.getElementById("image2imgSaveStatus").style.backgroundColor = '#840800';
				document.getElementById("image2imgSaveStatus").style.visibility='visible';
				break;
		}
		document.getElementById("image2imgGetVars").click();
		resetVariables();
		currGuide = Number(document.getElementById ("guidance_scale").value);
		oldGuide = Number(document.getElementById ("guidance_scale").value);
		currPrompt = Number(document.getElementById ("prompt_strength").value);
		oldPrompt = Number(document.getElementById ("prompt_strength").value);
		currLoRaModel = document.getElementById("lora_0").value;
		currLoRaStrength = Number(document.querySelector(".model_weight").value);
		oldLoRaStrength = Number(document.querySelector(".model_weight").value);
		currSeed = Number(document.getElementById ("seed").value);
		oldSeed = Number(document.getElementById ("seed").value);


		//console.log('Initial GS: ' + (Math.round(currGuide * 100) / 100).toFixed(2));
		//console.log('Initial PS: ' + (Math.round(currPrompt * 100) / 100).toFixed(2));
		//console.log('Initial Seed: ' + currSeed);

		myBatchCount = Number(document.getElementById ("image2imgBatchCount").value);

		myGuideStr = Number(document.getElementById ("image2imgAdjGuideStr").value);
		myGuideAdj = Number(document.getElementById ("image2imgAdjGuideVal").value);

		myPromptStr = Number(document.getElementById ("image2imgAdjPromptStr").value);
		myPromptAdj = Number(document.getElementById ("image2imgAdjPromptVal").value);
		myPromptReset = Number(document.getElementById ("image2imgResetPrompt").value);

		myLoRaStr = Number(document.getElementById ("image2imgAdjLoRa").value);
		myLoRaAdj = Number(document.getElementById ("image2imgAdjLoRaVal").value);

		mySeedStr = Number(document.getElementById ("image2imgAdjSeed").value);
		mySeedAdj = Number(document.getElementById ("image2imgAdjSeedVal").value);

		myWarpCount = Number(document.getElementById ("image2imgWarpEvery").value);

		myRotateCount = Number(document.getElementById ("image2imgRotateEvery").value);

		myPanCount = Number(document.getElementById ("image2imgPanEvery").value);

		myPromptChange = Number(document.getElementById ("image2imgChangePrompt").value);

		defaultTimeOut = Number(document.getElementById ("image2imgTimeOut").value);

		//adjustSeedStatus = document.getElementById("image2imgSeed").checked;
		//console.log('Seed Switch: ' + adjustSeedStatus);

		switchStatus = document.getElementById("image2imgSource").checked;
		//console.log('img2img Switch: ' + switchStatus);

		guideCount = 1;
		promptCount = 1;
		promptReset = 0;
		loraCount = 1;
		seedCount = 1;
		promptChangeCount = 1;
		image2imgTaskCount = 0;
		warpCount = 0;
		rotateCount = 0;
		panCount=0;


		for (let i = 0; i < myBatchCount; i++) {
			switch (stopBatch) {
				case true:
					break;
				case false:
					//checkTasks();
					switch (switchStatus) {
						case false:
							break;
						case true:
							//checkImg2Img();
							break;
						default:
							break;
					}
					timeoutWait();
					checkTasks();
			}
		}
		timeoutWait();
		resetVariables();
	}

	function resetVariables() {
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			//setTimeout(resetVariables, 100, img_remaining, 0)
			setTimeout(resetVariables, defaultTimeOut, img_remaining, 0)
		} else {
			console.log('Resetting variables');

			image2imgGSVal.value = (Math.round(oldGuide * 100) / 100).toFixed(2);
			image2imgGuideSlider.value = oldGuide*10;
			document.getElementById("guidance_scale").value = oldGuide;

			image2imgPSVal.value = (Math.round(oldPrompt * 100) / 100).toFixed(2);
			image2imgPromptSlider.value = oldPrompt*10;
			document.getElementById("prompt_strength").value = oldPrompt;

			document.getElementById("seed").value = oldSeed;
			document.getElementById("image2imgOutstanding").value = 0;
			switch (stopBatch) {
				case false:
					document.getElementById ("image2imgSaveStatus").value = "Batch Completed." ;
					break;
				case true:
					document.getElementById ("image2imgSaveStatus").value = "Batch stopped by User." ;
					break;
			}
			stopBatch = false;
		}
	}

	function timeoutWait() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			//setTimeout(resetVariables, 100, img_remaining, 0)
			setTimeout(timeoutWait, defaultTimeOut, img_remaining, 0)
		} else {
			var SomeVar = 0;
		}
	}

	function image2imgZeroVals() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			//setTimeout(resetVariables, 100, img_remaining, 0)
			setTimeout(image2imgZeroVals, defaultTimeOut, img_remaining, 0)
		} else {
			console.log('Setting all variables to Zero');
			document.getElementById ("image2imgAdjGuideStr").value = 0;
			document.getElementById ("image2imgAdjGuideVal").value = 0;
			document.getElementById ("image2imgResetGuide").value = 0;

			document.getElementById ("image2imgAdjPromptStr").value = 0;
			document.getElementById ("image2imgAdjPromptVal").value = 0;
			document.getElementById ("image2imgResetPrompt").value = 0;

			document.getElementById ("image2imgAdjLoRa").value = 0;
			document.getElementById ("image2imgAdjLoRaVal").value = 0;
			document.getElementById ("image2imgResetLora").value = 0;

			document.getElementById ("image2imgAdjSeed").value = 0;
			document.getElementById ("image2imgAdjSeedVal").value = 0;
			document.getElementById ("image2imgResetSeed").value = 0;

			document.getElementById ("image2imgWarpEvery").value = 0;
			document.getElementById ("image2imgWarpWidth").value = 0;
			document.getElementById ("image2imgWarpHeight").value = 0;

			document.getElementById ("image2imgRotateEvery").value = 0;
			document.getElementById ("image2imgRotateAngle").value = 0;

			document.getElementById ("image2imgPanEvery").value = 0;
			document.getElementById ("image2imgPanX").value = 0;
			document.getElementById ("image2imgPanY").value = 0;


		}
	}

	function checkTasks() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}

		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			//setTimeout(checkTasks, 100, img_remaining, 0)
			setTimeout(checkTasks, defaultTimeOut, img_remaining, 0)

		} else {
			image2imgTaskCount ++;
			console.log('Batch Loop: ' + image2imgTaskCount);
			switch (stopBatch) {
				case false:
					console.log('Starting next task');
					checkReadyStatus();
					//checkMadWarp(); //Link to Madrang Render Tasks plugin
					checkImg2Img();
					checkOptions();
					//makeWarpAdjustments();
					//makeRotateAdjustments();
					makeGuideAdjustments();
					makePromptAdjustments();
					makeLoRaAdjustments();
					makeSeedAdjustments();
					makePromptChange();
					console.log('Checking Status');
					showToast('Checking Status.', 1000, true);
					checkReadyStatus();
					showToast('Stable Diffusion is ready.', 2000, true);
					console.log('Creating image');

					document.getElementById("makeImage").click();
					updateCount();
					break;
				case true:
					break;
			}
		}
	}

	function checkImg2Img() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
			default:
				break;
		}
		switch (switchStatus) {
			case false:
				return;
				break;
			case true: // *******************************************************************************************************
				image2imgTest3(); // Remove Me
				return; // Remove me
				break;
			default:
				break;
		}
		console.log('Checking img2img');
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			//setTimeout(checkImg2Img, 100, img_remaining, 0)
			setTimeout(checkImg2Img, defaultTimeOut, img_remaining, 0)
		} else {
			//console.log(image2imgTaskCount);
			switch (image2imgTaskCount) {
				case 0:
					return;
					break;
				case 1:
					return;
					break;
				default:
					break;
			}
			console.log('Updating img2img');

			var imgElement = document.querySelectorAll('.imgContainer img');


			var imgLength = imgElement.length;
			var item = imgElement[0].src;

			var imgWidth = (imgElement[0].naturalWidth);
			var imgHeight =  (imgElement[0].naturalHeight);
			var imgIntrinsicWidth = (imgElement[0].width);
			var imgIntrinsicHeight = (imgElement[0].height);
			var imgRenderedWidth =imgWidth;
			var imgRenderedHeight =imgHeight;
			var imgMaxHeight = 150;
			var aspectRatio = imgWidth / imgHeight;

						//console.log("Intrinsic size: "+ imgIntrinsicWidth +"x"+ imgIntrinsicHeight);
						//console.log("Rendered size: "+ imgRenderedWidth +"x"+ imgRenderedHeight);
						//console.log('Aspect: ' + (Math.round(aspectRatio * 100) / 100).toFixed(2));

			const cnv = document.createElement('canvas'),
			ctx = cnv.getContext('2d', { willReadFrequently: true });
			ctx.save();
			cnv.width = imgWidth;
			cnv.height = imgHeight;
			ctx.drawImage(imgElement[0], 0, 0, imgWidth, imgHeight) // cnv.width, cnv.height);
			ctx.restore();
			const image = cnv.toDataURL('image/png');

			document.getElementById("init_image_preview").setAttribute("src",image);
			console.log(image);

			document.getElementById("image2imgTest").click();



		}
	}

	function checkOptions() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}

		switch (myWarpCount) {
			case 0:
				zoomFlag = false;
				break;
			default:
				zoomFlag = true;
				break;
		}

		switch (myRotateCount) {
			case 0:
				rotateFlag = false;
				break;
			default:
				rotateFlag = true;
				break;
		}

		switch (myPanCount) {
			case 0:
				panFlag = false;
				break;
			default:
				panFlag = true;
				break;
		}


		transitionOptions = "";

		switch (zoomFlag) {
			case true:
				transitionOptions = transitionOptions + 'Zoom';
				break;
			default:
				break;
		}

		switch (rotateFlag) {
			case true:
				transitionOptions = transitionOptions + 'Rotate';
				break;
			default:
				break;
		}

		switch (panFlag) {
			case true:
				transitionOptions = transitionOptions + 'Pan';
				break;
			default:
				break;
		}


		switch (transitionOptions) {
			case "Zoom":
				console.log('Transitions: ' + transitionOptions);
				makeWarpAdjustments();
				break;
			case "Rotate":
				console.log('Transitions: ' + transitionOptions);
				makeRotateAdjustments();
				break;
			case "Pan":
				console.log('Transitions: ' + transitionOptions);
				makePanAdjustments();
				break;
			case "ZoomRotate":
				console.log('Transitions: ' + transitionOptions);
				makeZoomAndRotateAdjustments();
				break;
			case "ZoomPan":
				console.log('Transitions: ' + transitionOptions);
				makeZoomAndPanAdjustments();
				break;
			case "RotatePan":
				console.log('Transitions: ' + transitionOptions);
				makeRotateAndPanAdjustments();
				break;
			case "ZoomRotatePan":
				console.log('Transitions: ' + transitionOptions);
				makeZoomRotateAndPanAdjustments();
				break;
			default:
				console.log('Transitions: None');
				break;
		}
	}



	function checkMadWarp() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		var MadtempVar = document.getElementById("MadWarpTest");
		switch (MadtempVar) {
			case null:
				return;
				break;
			default:
				break;
		}
		var MadImgCount = 0; // Number(document.getElementById("image2imgWarpEvery").value);

		switch (MadImgCount) {
			case 0:
				return;
				break;
			default:
				//document.getElementById("MadWarpTest").click();
				break;
		}

	}

	function makeWarpAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (myWarpCount) {
			case 0:
				return;
				break;
			default:
				break;
		}
		//console.log('Zoom Adjustments');
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeWarpAdjustments, defaultTimeOut, img_remaining, 0)
		} else {
			var WarpWidth = 0;
			var WarpHeight = 0;
			switch (warpCount) {
				case myWarpCount:
					warpCount = 1;
					WarpWidth = Number(document.getElementById("image2imgWarpWidth").value);
					WarpHeight = Number(document.getElementById("image2imgWarpHeight").value);
					break;
				default:
					warpCount ++;
					return;
					break;
			}
		}
		switch (WarpWidth) {
			case 0:
				switch (WarpHeight) {
					case 0:
					console.log('Zoom factors both zero');
					showToast('Zoom factors are both zero', 2000, true);
					return;
					break;
				default:
					break;
				}
			default:
				break;
		}
		console.log('Updating img2img');
		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		var item = imgElement[0].src;

		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight =  (imgElement[0].naturalHeight);
		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		const cnv = document.createElement('canvas'),
		ctx = cnv.getContext('2d', { willReadFrequently: true });
		ctx.save();
		cnv.width = imgWidth;
		cnv.height = imgHeight;

		ctx.scale(WarpWidth, WarpHeight);

		//ctx.transform(WarpWidth,0,0,WarpHeight,-(WarpWidth-1)*imgWidth/2,-(WarpHeight-1)*imgHeight/2)

		var x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
		var y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
		ctx.drawImage(imgElement[0], x, y);

		//ctx.drawImage(imgElement[0], 0, 0, imgWidth, imgHeight);

		ctx.restore();
		const image = cnv.toDataURL('image/png');

		document.getElementById("init_image_preview").setAttribute("src",image);
	}

	function makeRotateAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (myRotateCount) {
			case 0:
				return;
				break;
			default:
				break;
		}
		//console.log('Rotate Adjustments');
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeRotateAdjustments, defaultTimeOut, img_remaining, 0)
		} else {
			var rotateAngle = 0;
			switch (rotateCount) {
				case myRotateCount:
					rotateCount = 1;
					rotateAngle = Number(document.getElementById("image2imgRotateAngle").value);
					break;
				default:
					//console.log('Skipping');
					rotateCount ++;
					return;
					break;
			}
		}
		switch (rotateAngle) {
			case 0:
				console.log('Rotation angle is zero');
				showToast('Rotation angle is zero', 2000, true);
				return;
				break;
			default:
				break;
		}

		console.log('Updating img2img');
		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);


		const cnv = document.createElement('canvas'),
		ctx = cnv.getContext('2d', { willReadFrequently: true });

		ctx.save();
		cnv.width = imgWidth;
		cnv.height = imgHeight;
		//ctx.drawImage(imgElement[0], 0,0);
		ctx.translate(cnv.width/2, cnv.height/2)
		ctx.rotate(rotateAngle* Math.PI/180);
		ctx.translate(-(cnv.width/2), -(cnv.height/2));
		ctx.drawImage(imgElement[0], 0,0);

		ctx.restore();

		const image = cnv.toDataURL('image/png');

		document.getElementById("init_image_preview").setAttribute("src",image);

	}

	function makePanAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makePanAdjustments, defaultTimeOut, img_remaining, 0)
		}
		var makeModifications = false;
		var panChangeFlag = false;
		var PanX = 0;
		var PanY = 0;
		switch (panCount) {
			case myPanCount:
				panCount = 1;
				PanX = Number(document.getElementById("image2imgPanX").value);
				PanY = Number(document.getElementById("image2imgPanY").value);
				panChangeFlag = true;
				makeModifications = true;
				break;
			default:
				panCount ++;
				panChangeFlag = false;
				break;
		}
		switch (PanX) {
			case 0:
				switch (PanY) {
					case 0:
					console.log('Pan X and Y both zero');
					showToast('Pan X and Y both zero', 2000, true);
					return;
					break;
				default:
					break;
				}
			default:
				break;
		}
		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		switch (imgLength) {
			case 0:
				return;
				break;
			default:
				break;
		}
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);

		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		switch (makeModifications) {
			case true:
				const cnv = document.createElement('canvas'),
				ctx = cnv.getContext('2d', { willReadFrequently: true });
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				switch (panChangeFlag) {
					case true:
						ctx.save();
						ctx.translate(PanX, PanY);
						ctx.drawImage(imgElement[0],0,0);
						ctx.restore();
						const image = cnv.toDataURL('image/png');
						document.getElementById("init_image_preview").setAttribute("src",image);
						break;
					default:
						break;
				}
			default:
				break;
		}
	}


	function makeZoomAndRotateAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeZoomAndRotateAdjustments, defaultTimeOut, img_remaining, 0)
		}
		var makeModifications = false;
		var warpChangeFlag = false;
		var WarpWidth = 0;
		var WarpHeight = 0;
		switch (warpCount) {
			case myWarpCount:
				warpCount = 1;
				WarpWidth = Number(document.getElementById("image2imgWarpWidth").value);
				WarpHeight = Number(document.getElementById("image2imgWarpHeight").value);
				warpChangeFlag = true;
				makeModifications = true;
				break;
			default:
				warpCount ++;
				warpChangeFlag = false;
				break;
		}
		switch (WarpWidth) {
			case 0:
				switch (WarpHeight) {
					case 0:
						warpChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		var rotateAngle = 0;
		var rotateChangeFlag = false;

		switch (rotateCount) {
			case myRotateCount:
				rotateCount = 1;
				rotateAngle = Number(document.getElementById("image2imgRotateAngle").value);
				rotateChangeFlag = true;
				switch (rotateAngle) {
					case 0:
						rotateChangeFlag = false;
						break;
					default:
						break;
				}
				makeModifications = true;
				break;
			default:
				rotateCount ++;
				rotateChangeFlag = false;
				break;
		}

		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		switch (imgLength) {
			case 0:
				return;
				break;
			default:
				break;
		}
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);

		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		switch (warpChangeFlag) {
			case false:
				switch (rotateChangeFlag) {
					case false:
						makeModifications = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		console.log('Zoom: ' + warpChangeFlag);
		console.log('Rotate: ' + rotateChangeFlag);
		console.log('Modify: ' +  makeModifications);

		var x=0;
		var y=0;

		switch (makeModifications) {
			case true:
				const cnv = document.createElement('canvas'),
				ctx = cnv.getContext('2d', { willReadFrequently: true });

				ctx.save();
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				switch (warpChangeFlag) {
					case true:
						console.log('Zooming');
						ctx.scale(WarpWidth, WarpHeight);
						x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
						y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
						ctx.drawImage(imgElement[0], x, y);
						break;
					default:
						break;
				}

				switch (rotateChangeFlag) {
					case true:
						console.log('Rotating');
						ctx.translate(cnv.width/2, cnv.height/2)
						ctx.rotate(rotateAngle* Math.PI/180);
						ctx.translate(-(cnv.width/2), -(cnv.height/2));
						switch (WarpWidth) {
							case 0:
								x = (ctx.canvas.width  - imgWidth) * 0.5;
								break;
							default:
								x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
								break;
						}

						switch (WarpHeight) {
							case 0:
								y = (ctx.canvas.height - imgHeight) * 0.5;
								break;
							default:
								y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
								break;
						}

						ctx.drawImage(imgElement[0],x, y);
						break;
					default:
						break;
				}
				ctx.restore();
				const image = cnv.toDataURL('image/png');
				document.getElementById("init_image_preview").setAttribute("src",image);
				break;
			default:
				break;
		}
	}

	function makeZoomAndPanAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeZoomAndPanAdjustments, defaultTimeOut, img_remaining, 0)
		}
		var makeModifications = false;
		var warpChangeFlag = false;
		var WarpWidth = 0;
		var WarpHeight = 0;
		switch (warpCount) {
			case myWarpCount:
				warpCount = 1;
				WarpWidth = Number(document.getElementById("image2imgWarpWidth").value);
				WarpHeight = Number(document.getElementById("image2imgWarpHeight").value);
				warpChangeFlag = true;
				makeModifications = true;
				break;
			default:
				warpCount ++;
				warpChangeFlag = false;
				break;
		}

		switch (WarpWidth) {
			case 0:
				switch (WarpHeight) {
					case 0:
						warpChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		var PanX = 0;
		var PanY = 0;
		var panChangeFlag = false;

		switch (panCount) {
			case myPanCount:
				panCount = 1;
				PanX = Number(document.getElementById("image2imgPanX").value);
				PanY = Number(document.getElementById("image2imgPanY").value);
				panChangeFlag = true;
				makeModifications = true;
				break;
			default:
				panCount ++;
				panChangeFlag = false;
				break;
		}

		switch (PanX) {
			case 0:
				switch (PanY) {
					case 0:
						panChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}


		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		switch (imgLength) {
			case 0:
				return;
				break;
			default:
				break;
		}
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);

		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		switch (warpChangeFlag) {
			case false:
				switch (panChangeFlag) {
					case false:
						makeModifications = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		console.log('Zoom: ' + warpChangeFlag);
		console.log('Pan: ' + panChangeFlag);
		console.log('Modify: ' +  makeModifications);
		switch (makeModifications) {
			case true:
				const cnv = document.createElement('canvas'),
				ctx = cnv.getContext('2d', { willReadFrequently: true });
				ctx.save();
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				switch (warpChangeFlag) {
					case true:
						console.log('Zooming');
						ctx.scale(WarpWidth, WarpHeight);
						var x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
						var y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
						ctx.drawImage(imgElement[0], x, y);
						break;
					default:
						break;
				}

				switch (panChangeFlag) {
					case true:
						console.log('Panning');
						ctx.translate(PanX, PanY);
						ctx.drawImage(imgElement[0],0,0);
						break;
					default:
						break;
				}
				ctx.restore();
				const image = cnv.toDataURL('image/png');
				document.getElementById("init_image_preview").setAttribute("src",image);
				break;
			default:
				break;
		}
	}

	function makeZoomRotateAndPanAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeZoomRotateAndPanAdjustments, defaultTimeOut, img_remaining, 0)
		}
		var makeModifications = false;
		var warpChangeFlag = false;
		var WarpWidth = 0;
		var WarpHeight = 0;

		switch (warpCount) {
			case myWarpCount:
				warpCount = 1;
				WarpWidth = Number(document.getElementById("image2imgWarpWidth").value);
				WarpHeight = Number(document.getElementById("image2imgWarpHeight").value);
				warpChangeFlag = true;
				makeModifications = true;
				break;
			default:
				warpCount ++;
				warpChangeFlag = false;
				break;
		}

		switch (WarpWidth) {
			case 0:
				switch (WarpHeight) {
					case 0:
						warpChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}


		var rotateChangeFlag = false;
		var rotateAngle = 0;

		switch (rotateCount) {
			case myRotateCount:
				rotateCount = 1;
				rotateAngle = Number(document.getElementById("image2imgRotateAngle").value);
				rotateChangeFlag = true;
				makeModifications = true;
				break;
			default:
				rotateCount ++;
				rotateChangeFlag = false;
				break;
		}

		switch (rotateAngle) {
			case 0:
				rotateChangeFlag = false;
				break;
			default:
				break;
		}

		var panChangeFlag = false;
		var PanX = 0;
		var PanY = 0;

		switch (panCount) {
			case myPanCount:
				panCount = 1;
				PanX = Number(document.getElementById("image2imgPanX").value);
				PanY = Number(document.getElementById("image2imgPanY").value);
				panChangeFlag = true;
				makeModifications = true;
				break;
			default:
				panCount ++;
				panChangeFlag = false;
				break;
		}

		switch (PanX) {
			case 0:
				switch (PanY) {
					case 0:
						panChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		switch (imgLength) {
			case 0:
				return;
				break;
			default:
				break;
		}
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);

		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		switch (warpChangeFlag) {
			case false:
				switch (rotateChangeFlag) {
					case false:
						switch (panChangeFlag) {
							case false:
								makeModifications = false;
								break;
							default:
								break;
						}
					default:
						break;
				}
			default:
				break;
		}

		console.log('Zoom: ' + warpChangeFlag);
		console.log('Rotate: ' + rotateChangeFlag);
		console.log('Pan: ' + panChangeFlag);
		console.log('Modify: ' +  makeModifications);

		var x=0;
		var y=0;

		switch (makeModifications) {
			case true:
				const cnv = document.createElement('canvas'),
				ctx = cnv.getContext('2d', { willReadFrequently: true });
				ctx.save();
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				switch (warpChangeFlag) {
					case true:
						console.log('Zooming');
						ctx.scale(WarpWidth, WarpHeight);
						x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
						y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
						ctx.drawImage(imgElement[0], x, y);
						break;
					default:
						break;
				}
				switch (rotateChangeFlag) {
					case true:
						console.log('Rotating');
						ctx.translate(cnv.width/2, cnv.height/2)
						ctx.rotate(rotateAngle* Math.PI/180);
						ctx.translate(-(cnv.width/2), -(cnv.height/2));
						switch (WarpWidth) {
							case 0:
								x = (ctx.canvas.width  - imgWidth) * 0.5;
								break;
							default:
								x = (ctx.canvas.width / WarpWidth - imgWidth) * 0.5;
								break;
						}

						switch (WarpHeight) {
							case 0:
								y = (ctx.canvas.height - imgHeight) * 0.5;
								break;
							default:
								y = (ctx.canvas.height / WarpHeight - imgHeight) * 0.5;
								break;
						}

						ctx.drawImage(imgElement[0], x, y);
						break;
					default:
						break;
				}
				switch (panChangeFlag) {
					case true:
						console.log('Panning');
						ctx.translate(PanX, PanY);
						ctx.drawImage(imgElement[0],0,0);
						break;
					default:
						break;
				}
				ctx.restore();
				const image = cnv.toDataURL('image/png');
				document.getElementById("init_image_preview").setAttribute("src",image);
				break;
			default:
				break;
		}
	}


	function makeRotateAndPanAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(makeRotateAndPanAdjustments, defaultTimeOut, img_remaining, 0)
		}
		var makeModifications = false;
		var rotateChangeFlag = false;
		var rotateAngle = 0;
		switch (rotateCount) {
			case myRotateCount:
				rotateCount = 1;
				rotateAngle = Number(document.getElementById("image2imgRotateAngle").value);
				rotateChangeFlag = true;
				makeModifications = true;
				break;
			default:
				rotateCount ++;
				rotateChangeFlag = false;
				break;
		}

		switch (rotateAngle) {
			case 0:
				rotateChangeFlag = false;
				break;
			default:
				break;
		}

		var panChangeFlag = false;
		var PanX = 0;
		var PanY = 0;

		switch (panCount) {
			case myPanCount:
				panCount = 1;
				PanX = Number(document.getElementById("image2imgPanX").value);
				PanY = Number(document.getElementById("image2imgPanY").value);
				panChangeFlag = true;
				makeModifications = true;
				break;
			default:
				panCount ++;
				panChangeFlag = false;
				break;
		}

		switch (PanX) {
			case 0:
				switch (PanY) {
					case 0:
						panChangeFlag = false;
						break;
					default:
						break;
				}
			default:
				break;
		}

		var imgElement = document.querySelectorAll('.imgContainer img');
		var imgLength = imgElement.length;
		switch (imgLength) {
			case 0:
				return;
				break;
			default:
				break;
		}
		var item = imgElement[0].src;
		var imgWidth = (imgElement[0].naturalWidth);
		var imgHeight = (imgElement[0].naturalHeight);

		var imgIntrinsicWidth = (imgElement[0].width);
		var imgIntrinsicHeight = (imgElement[0].height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		switch (rotateChangeFlag) {
			case false:
				switch (panChangeFlag) {
					case false:
						makeModifications = false;
						break;
					default:
						break;
				}
			default:
			break;
		}

		console.log('Rotate: ' + rotateChangeFlag);
		console.log('Pan: ' + panChangeFlag);
		console.log('Modify: ' +  makeModifications);

		switch (makeModifications) {
			case true:
				const cnv = document.createElement('canvas'),
				ctx = cnv.getContext('2d', { willReadFrequently: true });
				ctx.save();
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				switch (rotateChangeFlag) {
					case true:
						console.log('Rotating');
						ctx.translate(cnv.width/2, cnv.height/2)
						ctx.rotate(rotateAngle* Math.PI/180);
						ctx.translate(-(cnv.width/2), -(cnv.height/2));
						ctx.drawImage(imgElement[0], 0, 0);
						break;
					default:
						break;
				}
				switch (panChangeFlag) {
					case true:
						console.log('Rotating');
						ctx.translate(PanX, PanY);
						ctx.drawImage(imgElement[0],0,0);
						break;
					default:
						break;
				}
				ctx.restore();
				const image = cnv.toDataURL('image/png');
				document.getElementById("init_image_preview").setAttribute("src",image);
				break;
			default:
				break;
		}
	}



	function makeGuideAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (myGuideStr) {
			case 0:
				return;
				break;
			default:
				break;
		}
		switch (guideCount) {
			case myGuideStr:
				//console.log('Making GS adjustments');
				guideCount = 1;
				currGuide = currGuide + myGuideAdj;

				if (currGuide < minGuide) {
					currGuide = minGuide;
				} else if (currGuide > maxGuide) {
					currGuide = maxGuide;
				}

				image2imgGSVal.value = (Math.round(currGuide * 100) / 100).toFixed(2);
				document.getElementById("guidance_scale").value = image2imgGSVal.value;
				image2imgGuideSlider.value = currGuide*10;
				//console.log('GS: ' + (Math.round(currGuide * 100) / 100).toFixed(2));
				break;
			default:
				image2imgGSVal.value = (Math.round(currGuide * 100) / 100).toFixed(2);
				image2imgGuideSlider.value = currGuide*10;
				guideCount ++;
				break;
		}
	}

	function makePromptAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (myPromptStr) {
			case 0:
				return;
				break;
			default:
				break;
		}
		switch (promptCount) {
			case myPromptStr:
				//console.log('Making PS adjustments');
				promptCount = 1;
				currPrompt = currPrompt + myPromptAdj;
				if (currPrompt < minPrompt) {
					currPrompt = minPrompt;
				} else if (currPrompt > maxPrompt) {
					currPrompt = maxPrompt;
				}
				image2imgPSVal.value = (Math.round(currPrompt * 100) / 100).toFixed(2);
				document.getElementById("prompt_strength").value = image2imgPSVal.value;
				image2imgPromptSlider.value = currPrompt*10;
				//console.log('PS: ' + (Math.round(currPrompt * 100) / 100).toFixed(2));
				break;
			default:
				image2imgPSVal.value = (Math.round(currPrompt * 100) / 100).toFixed(2);
				image2imgPromptSlider.value = currPrompt*10;
				promptCount ++;
				break;
			}
		//myPromptReset = Number(document.getElementById ("image2imgResetPrompt").value);

		switch (promptReset) {
			case (myPromptReset):
				console.log('Resetting PS');
				promptReset = 1;
				currPrompt = oldPrompt;
				if (currPrompt < minPrompt) {
					currPrompt = minPrompt;
				} else if (currPrompt > maxPrompt) {
					currPrompt = maxPrompt;
				}
				image2imgPSVal.value = (Math.round(currPrompt * 100) / 100).toFixed(2);
				document.getElementById("prompt_strength").value = image2imgPSVal.value;
				image2imgPromptSlider.value = currPrompt*10;
				break;
			default:
				image2imgPSVal.value = (Math.round(currPrompt * 100) / 100).toFixed(2);
				image2imgPromptSlider.value = currPrompt*10;
				promptReset ++;
				break;
		}

	}

	function makeLoRaAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (myLoRaStr) {
			case 0:
				return;
				break;
			default:
				break;
		}
		currLoRaModel = document.getElementById("lora_0").value;
		switch (currLoRaModel) {
			case 'None':
				return;
				break;
			default:
				break;
		}

		switch (loraCount) {
			case myLoRaStr:
				loraCount = 1;
				currLoRaStrength = currLoRaStrength + myLoRaAdj;
				if (currLoRaStrength < minLoRa) {
					currLoRaStrength = minLoRa;
				} else if (currLoRaStrength > maxLoRa) {
					currLoRaStrength = maxLoRa;
				}
				image2imgLoRaVal.value = (Math.round(currLoRaStrength * 100) / 100).toFixed(2);
				document.getElementById("image2imgLoRaVal").value = image2imgLoRaVal.value;
				var newSetVal = parseFloat(document.getElementById("image2imgLoRaVal").value);
				var tempModel = document.getElementsByClassName("model_entry");
				tempModel.item(0).querySelector(".model_weight").value = newSetVal;
				break;
			default:
				image2imgLoRaVal.value = (Math.round(currLoRaStrength * 100) / 100).toFixed(2);
				loraCount ++;
				break;
			}

	}



	function makeSeedAdjustments() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		switch (adjustSeedStatus) {
			case false:
				//console.log('Not adjusting Seed');
				return;
				break;
			default:
				//console.log('Adjusting Seed');
				break;
		}
		switch (seedCount) {
			case mySeedStr:
				//console.log('Making Seed adjustments');
				seedCount = 1;
				currSeed = currSeed + mySeedAdj;
				if (currSeed < minSeed) {
					currSeed = minSeed;
				}
				document.getElementById("seed").value = currSeed;
				//console.log('Seed: ' + currSeed);
				break;
			default:
				document.getElementById("seed").value = currSeed;
				seedCount ++;
				break;
			}
	}

	function image2imgGetVars() {
		var DummyGuide = document.getElementById("guidance_scale").value;
		var DummyPrompt = document.getElementById("prompt_strength").value;
		var DummySeed = document.getElementById("seed").value;
		oldGuide = DummyGuide;
		oldPrompt = DummyPrompt;
		currSeed = DummySeed;
		oldSeed = DummySeed;
	}

	function image2imgStopBatch() {
		console.log('Stopping batch generation');
		document.getElementById("stopImage").click();
		stopBatch = true;
		document.getElementById("image2imgOutstanding").value =0;
		document.getElementById ("image2imgSaveStatus").value = "Stopping Batch." ;
	}

	function updateCount() {
		var imgLeftToProcess = (myBatchCount - image2imgTaskCount)+1;
		document.getElementById("image2imgOutstanding").value = imgLeftToProcess;
		var myTempVar = getSetting('save_to_disk');
		var myTempVar2 = document.getElementById("preview-tools");
		switch (myTempVar) {
			case true:
				document.getElementById ("image2imgSaveStatus").value = "Save Status is On. (" + imgLeftToProcess + " Images Left)" ;
				document.getElementById("image2imgSaveStatus").style.backgroundColor = '#13750d';
				document.getElementById("image2imgSaveStatus").style.visibility='visible';
				break;
			case false:
				document.getElementById("image2imgSaveStatus").value = "Save Status is Off. (" + imgLeftToProcess + " Images Left)";
				document.getElementById("image2imgSaveStatus").style.backgroundColor = '#840800';
				document.getElementById("image2imgSaveStatus").style.visibility='visible';
				break;
		}

	}

	function imageBuildScene() {
		if (document.getElementById("image2imgBuildScene")){
			var DummyStatus = true;
		} else {
			document.getElementById("image2imgBuildScene").checked = false;
		}

		BuildSceneStatus = document.getElementById("image2imgBuildScene").checked;
		switch (BuildSceneStatus) {
			case true:
				document.getElementById("image2imgDressMeUp").checked = false;
				document.getElementById("image2imgText2Prompt").checked = false;
				document.getElementById("image2imgSeqText2Prompt").checked = false;
				//BuildSceneStatus = false;
				DressMeUpStatus = false;
				Text2PromptStatus = false;
				Text2PromptSeqStatus = false;
				break;
			case false:
				break;
		}
		console.log('BuildScene: ' + BuildSceneStatus);
	}

	function imageDressMeUp() {
		if (document.getElementById("image2imgDressMeUp")){
			var DummyStatus = true;
		} else {
			document.getElementById("image2imgDressMeUp").checked = false;
		}

		DressMeUpStatus = document.getElementById("image2imgDressMeUp").checked;
		switch (DressMeUpStatus) {
			case true:
				document.getElementById("image2imgBuildScene").checked = false;
				document.getElementById("image2imgText2Prompt").checked = false;
				document.getElementById("image2imgSeqText2Prompt").checked = false;
				BuildSceneStatus = false;
				//DressMeUpStatus = false;
				Text2PromptStatus = false;
				Text2PromptSeqStatus = false;
				break;
			case false:
				break;
		}
		console.log('DressMeUp: ' + DressMeUpStatus);

	}
	function imageText2Prompt() {
		if (document.getElementById("image2imgText2Prompt")){
			var DummyStatus = true;
		} else {
			document.getElementById("image2imgText2Prompt").checked = false;
		};

		Text2PromptStatus = document.getElementById("image2imgText2Prompt").checked;
		switch (Text2PromptStatus) {
			case true:
				document.getElementById("image2imgBuildScene").checked = false;
				document.getElementById("image2imgDressMeUp").checked = false;
				document.getElementById("image2imgSeqText2Prompt").checked = false;
				BuildSceneStatus = false;
				DressMeUpStatus = false;
				//Text2PromptStatus = false;
				Text2PromptSeqStatus = false;
				break;
			case false:
				break;
		}
		console.log('Text2Prompt: ' + Text2PromptStatus);
	}

	function imageText2PromptSeq() {
		if (document.getElementById("image2imgSeqText2Prompt")){
			var DummyStatus = true;
		} else {
			document.getElementById("image2imgSeqText2Prompt").checked = false;
		};
		Text2PromptSeqStatus = document.getElementById("image2imgSeqText2Prompt").checked;
		switch (Text2PromptSeqStatus) {
			case true:
				document.getElementById("image2imgBuildScene").checked = false;
				document.getElementById("image2imgDressMeUp").checked = false;
				document.getElementById("image2imgText2Prompt").checked = false;
				BuildSceneStatus = false;
				DressMeUpStatus = false;
				Text2PromptStatus = false;
				//Text2PromptSeqStatus = false;
				break;
			case false:
				break;
		}
		console.log('Text2PromptSeq: ' + Text2PromptSeqStatus);
	}

	function makePromptChange() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}

		switch (image2imgTaskCount) {
			case 1:
				//console.log('First Image');
				promptChangeCount = myPromptChange;
				break;
			default:
				break;
		}

		switch (promptChangeCount) {
			case myPromptChange:
				console.log('Changing Prompt');
				var dummyToDo = Number(document.getElementById("image2imgOutstanding").value);
				switch (dummyToDo) {
					case 0:
						showToast('Setting initial Prompt.', 2000, true);
						break;
					default:
						showToast('Changing Prompt.', 1000, true);
						showToast((dummyToDo - 1) + ' images left to process.', 4000, true);
						break;
				}
				promptChangeCount = 0;
				switch (BuildSceneStatus) {
					case true:
						console.log('Build-A-Scene');
						document.getElementById("setRandomScene").click();
						document.getElementById("replaceScenePrompt").click();
						break;
					default:
						break;
				}
				switch (DressMeUpStatus) {
					case true:
						console.log('DressMeUp');
						document.getElementById("setRandomItems").click();
						document.getElementById("setPrompt").click();
						break;
					default:
						break;
				}
				switch (Text2PromptStatus) {
					case true:
						console.log('Random Wildcard');
						document.getElementById("setRandomText2Prompt").click();
						document.getElementById("setText2Prompt").click();
						break;
					default:
						break;
				}
				switch (Text2PromptSeqStatus) {
					case true:
						console.log('Sequential Wildcard');
						switch (firstWildcardPass) {
							case true:
								document.getElementById("setFirstText2Prompt").click();
								firstWildcardPass = false;
								break;
							case false:
								document.getElementById("setNextText2Prompt").click();
								break;
						}
						break;
					default:
						break;
				}
			default:
				//console.log('Breaking');
				break;
		}
		promptChangeCount ++;
	}


	function image2imgTest() {
		console.log('Test Routine');
		timeoutWait();
		checkReadyStatus();
		let iLoop=0
		document.querySelectorAll(".imageTaskContainer").forEach(container => {
			let req = htmlTaskMap.get(container)
			container.querySelectorAll(".imgContainer img").forEach(img => {
				if (img.closest('.imgItem').style.display === 'none') {
					//console.log('skipping hidden image', img)
					return
				}

            setTimeout(() => {image2imgTest2(req, img,iLoop)}, iLoop*200)
				iLoop=iLoop+1
            // console.log(req)
			})
		})
	}

	function image2imgTest2(req, img,iLoop) {
		console.log('Test2');
		timeoutWait();
		checkReadyStatus();
		// console.log(img);
		var MyimgCounter = (Number(img.getAttribute("data-imagecounter")));
		var myTempCount = Number(image2imgTaskCount);
		console.log('MyCounter: ' + MyimgCounter);
		switch (MyimgCounter) {
			case myTempCount:
				console.log('Batch Match');
				console.log(img);
				break;
			default:
				//console.log('No Match');
				return;
				break;
		}
		var imgWidth = (img.naturalWidth);
		var imgHeight =  (img.naturalHeight);
		var imgIntrinsicWidth = (img.width);
		var imgIntrinsicHeight = (img.height);
		var imgRenderedWidth =imgWidth;
		var imgRenderedHeight =imgHeight;
		var imgMaxHeight = 150;
		var aspectRatio = imgWidth / imgHeight;

		//const cnv = document.createElement('canvas'),
			//ctx = cnv.getContext('2d');
			//ctx.save();
			//cnv.width = imgWidth;
			//cnv.height = imgHeight;
			//ctx.drawImage(imgElement[0], 0, 0, imgWidth, imgHeight) // cnv.width, cnv.height);
			//ctx.restore();
			//const image = cnv.toDataURL('image/png');

			//document.getElementById("init_image_preview").setAttribute("src",image);



		const cnv = document.createElement('canvas');
		var ctx = cnv.getContext('2d', { willReadFrequently: true });
		ctx.save();
		cnv.width = imgWidth;
		cnv.height = imgHeight;
		ctx.drawImage(img, 0, 0, imgWidth, imgHeight) // cnv.width, cnv.height);
		ctx.restore();
		const image = cnv.toDataURL('image/png');
		document.getElementById("init_image_preview").setAttribute("src",image);
	}

	function image2imgTest3() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
			default:
				break;
		}
		//timeoutWait();
		//checkReadyStatus();

		let all_tasks = [...document.querySelectorAll("div .imageTaskContainer").entries()].map(c => htmlTaskMap.get(c[1]))
		let tasks_to_be_run = all_tasks.filter(task => task.isProcessing)
		let img_remaining_per_task = tasks_to_be_run.map(task => { return task.numOutputsTotal - Math.max(0, (task.instances || []).reduce((total, value) => {
			if (value.status === 'completed') {
				return total + 1
			} else {
				return total
			}
		}, 0) * task.reqBody.num_outputs) } )
		let img_remaining = img_remaining_per_task.reduce((total, value) => total + value, 0);
		if (img_remaining > 0) {
			setTimeout(image2imgTest3, defaultTimeOut, img_remaining, 0)
		} else {
			console.log('Checking img2img');
			document.getElementById ("image2imgSaveStatus").value = "Waiting for img2img to complete" ;
		}
		//var ReadyStatusFlag = true;
		//var ReadyStatus = document.querySelector("#server-status").querySelector("#server-status-msg").innerHTML;
		//if (ReadyStatus !="Stable Diffusion is ready") {
		//	ReadyStatusFlag = false;
		//} else {
		//	ReadyStatusFlag = true;
		//}
		//if (ReadyStatusFlag != true) {
		//	setTimeout(image2imgTest3, defaultTimeOut);
		//}

		var imgFoundFlag = false;
		var myTempImg = null;

		document.querySelectorAll(".imageTaskContainer").forEach((container) => {
			container.querySelectorAll(".imgContainer img").forEach((img) => {
				let imgItem = img.closest(".imgItem");
				if (imgItem.style.display === "none") {
					return
				}
				let req = imageRequest[img.dataset["imagecounter"]]
				switch (imgFoundFlag) {
					case false:
						//console.log(imgItem);
						timeoutWait();
						checkReadyStatus();
						timeoutWait();
						checkReadyStatus();
						myTempImg = imgItem.getElementsByTagName('img')[0];
						//console.log('Image: ' + myTempImg);
						imgFoundFlag= true;
						break;
					case true:
						break;
				}
			})
		})

		switch (imgFoundFlag) {
			case false:
				return;
				break;
			case true:
				//console.log('Image: ' + myTempImg);
				var imgWidth = (myTempImg.naturalWidth);
				var imgHeight =  (myTempImg.naturalHeight);
				var imgIntrinsicWidth = (myTempImg.width);
				var imgIntrinsicHeight = (myTempImg.height);
				var imgRenderedWidth =imgWidth;
				var imgRenderedHeight =imgHeight;
				var imgMaxHeight = 150;
				var aspectRatio = imgWidth / imgHeight;
				const cnv = document.createElement('canvas');
				var ctx = cnv.getContext('2d', { willReadFrequently: true });
				//const ctx = canvas.getContext('2d', { willReadFrequently: true });
				ctx.save();
				cnv.width = imgWidth;
				cnv.height = imgHeight;
				ctx.drawImage(myTempImg, 0, 0, imgWidth, imgHeight) // cnv.width, cnv.height);
				ctx.restore();
				const image = cnv.toDataURL('image/png');
				switch (image.length) {
					case 0:
						console.log('Image Error');
						break;
					case 6:
						console.log('Image Error');
						break;
					default:
						document.getElementById("init_image_preview").setAttribute("src",image);
						break;
				}
				//console.log(image);
				break;
		}

		//var containerMatch = false;
		//document.querySelectorAll(".imageTaskContainer").forEach(container => {
		//	checkReadyStatus();
		//	let req = htmlTaskMap.get(container);
		//	container.querySelectorAll(".imgContainer img").forEach(img => {
		//		checkReadyStatus();
		//		var MyimgCounter = (Number(img.getAttribute("data-imagecounter")));
		//		if (img.closest('.imgItem').style.display === 'none') {
		//			containerMatch = false;
		//		} else {
		//			containerMatch = true;
		//		}
		//		console.log('Container: ' + containerMatch);
		//		console.log('***');
		//		switch (MyimgCounter) {
		//			case 0:
		//				console.log('Counter: ' + MyimgCounter);
		//				console.log(img.src);
		//				var imgWidth = (img.naturalWidth);
		//				var imgHeight =  (img.naturalHeight);
		//				var imgIntrinsicWidth = (img.width);
		//				var imgIntrinsicHeight = (img.height);
		//				var imgRenderedWidth =imgWidth;
		//				var imgRenderedHeight =imgHeight;
		//				var imgMaxHeight = 150;
		//				var aspectRatio = imgWidth / imgHeight;
		//				const cnv = document.createElement('canvas');
		//				var ctx = cnv.getContext('2d');
		//				ctx.save();
		//				cnv.width = imgWidth;
		//				cnv.height = imgHeight;
		//				ctx.drawImage(img, 0, 0, imgWidth, imgHeight) // cnv.width, cnv.height);
		//				ctx.restore();
		//				const image = cnv.toDataURL('image/png');
		//				document.getElementById("init_image_preview").setAttribute("src",image);
		//				break;
		//			default:
		//				//console.log('*** Error ***');
		//				//console.log(img);
		//				break;
		//		}
		//
		//	})
		//})
	}


	function checkReadyStatus() {
		switch (stopBatch) {
			case true:
				return;
				break;
			case false:
				break;
		}
		var ReadyStatusFlag = true;
		var ReadyStatus = document.querySelector("#server-status").querySelector("#server-status-msg").innerHTML;
		if (ReadyStatus !="Stable Diffusion is ready") {
			ReadyStatusFlag = false;
		} else {
			ReadyStatusFlag = true;
		}
		if (ReadyStatusFlag != true) {
			setTimeout(checkReadyStatus, defaultTimeOut);
		}
	}



})();
