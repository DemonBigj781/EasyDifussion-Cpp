/**
 * Text to Prompt
 * v.1.1, last updated: 04/01/2024
 * By The Stig
 *
 * Change Log
 * 04/01/2024 Removed the Generate Single Image (user request)
 * 27/12/2023 Set Default State to Closed
 * 17/11/2023 Added Clear Search Function
 * 16/11/2023 Added Search Function (experimental)
 * 14/11/2023 Fixed bug to remove double spaces
 * 14/11/2023 Added Get Previous Prompt button to get the Previous entry in the Wildcard file
 * 14/11/2023 Added Get Last Prompt button to get the Last entry in the Wildcard file
 * 13/11/2023 Fixed bug to allow duplicate entries
 * 11/11/2023 Added Get First Prompt button to get the First entry in the Wildcard file
 * 11/11/2023 Added Get Next Prompt button to get the Next entry in the Wildcard file
 * 10/11/2023 Added Generate Sequential Batch button to generate all images in Wildcard file
 * 10/11/2023 Added code to allow integration with Image img2img Plugin
 * 08/11/2023 Initial Build
 *
 * Free to use with the CMDR2 Stable Diffusion UI.
 *
 */
(function() { "use strict"
    const VERSION = "1.1";
    const ID_PREFIX = "TheStig-Text2Prompt-plugin";
    console.log('%s Embed Metadata Version: %s', ID_PREFIX, VERSION);

	injectLoaderCSS();
	addBuildOptions();

	var text2promptArray=[];
	var text2promptPosition = 0;

	var text2PromptInput=null;
	var myCheckFlag = false;


	function injectLoaderCSS() {
		console.log('** Inject CSS **');
		const style = document.createElement('style');
		style.textContent = `
			.Text2Prompt {
				background-color:#2168bf;
			}
			.Text2Prompt2 {
				background-color:#13750d;
			}
			.Text2Prompt3 {
				background-color:#840800;
			}
			.inputWidth {
				width:280px;
				min-width: 280px;
				max-width: 280px ;
				overflow: hidden;
			}

			.WildcardList {
				width:280px;
				min-width: 280px;
				max-width: 280px ;
				overflow: hidden;
			}


			.labelWidth {
				width:64px  !important;
				display: inline-block;
				overflow: hidden;
			}
			.labelWidth2 {
				width:160px  !important;
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
			.valid {
				border: 3px solid green;
			}
			.invalid {
				border: 3px solid red;
			}
		`;
		document.head.appendChild(style);
	}

	function addBuildOptions() {
		console.log('Add Build a Scene Settings');
		var openText2PromptCheck = '';
		//openText2PromptCheck = ' active';
		var BuildSettings = document.createElement('div');
		BuildSettings.id = 'BuildAScene-settings';
		BuildSettings.classList.add('panel-box');
		let tempHTML = `
			<h4 class="collapsible `+ openText2PromptCheck +`">Wildcard Text to Prompt</h4>
			<div id="t2p-settings-entries" class="collapsible-content" style="display: block;margin-top:15px;">
				<button style="display:block;width:240px; height:30px;" class = "Text2Prompt" onclick="document.getElementById('Text2PromptSelectFiles').click()">Import Wildcard Text File</button>
				<input type='file' id="Text2PromptSelectFiles" style="display:none" onchange="document.getElementById('importText2File').click()">
				<input type="file" id="Text2PromptPicker" name="fileList" style="display:none" webkitdirectory multiple />
				<button style="display:none" type="button" id="locateText2File">Locate Text Files</button>
				<button style="display:none" type="button" id="importText2File">Import Scene</button>
				<p></p>
				<label for="text2promptCount">Number of Prompts</label>
				<input type="number" title="Number of Prompts" id="text2promptCount" value="0">
				<p></p>
				<label for="text2prompt_Options" class="labelWidth">Prompt</label>
				<select id="text2prompt_Options" class="inputWidth" name="text2prompt_Options onchange = "selectOption()">
				<option>None</option></select>
				<p></P>
				<div>
					<datalist id="WildcardList">
						<option>No Wildcard file currently loaded</option>
					</datalist>
					<label for="text2prompt_Options2" class="labelWidth">Search</label>
					<input  autoComplete="on" id="text2prompt_Options2" class="WildcardList" list="WildcardList"/>
				</div>
				<p></p>
				<button type="button" class = "Text2Prompt2" id="text2PromptUseSearch">Use Search Result</button>
				<p></p>
				<button type="button" class = "Text2Prompt3" id="text2PromptClearSearch">Clear Search Result</button>
				<p></p>
				<label for="preText2Prompt">Pre Prompt:</label>
				<textarea title="Text before the main prompt (editable)" class="txtBox2" id="preText2Prompt" name="preText2Prompt" rows="2" cols="60"></textarea>
				<p></p>
				<label for="currText2Prompt">Main Prompt:</label>
				<textarea readonly class="txtBox2" id="currText2Prompt" name="currText2Prompt" rows="6" cols="60"></textarea>
				<p></p>
				<label for="postText2Prompt">Post Prompt:</label>
				<textarea title="Text after the main prompt (editable)" class="txtBox2" id="postText2Prompt" name="postText2Prompt" rows="2" cols="60"></textarea>
				<p></p>
				<button type="button" class = "Text2Prompt2" id="setFirstText2Prompt">Get the First Prompt</button>
				<p></p>
				<button type="button" class = "Text2Prompt2" id="setNextText2Prompt">Get the Next Prompt</button>
				<p></p>
				<button type="button" class = "Text2Prompt2" id="setPrevText2Prompt">Get the Previous Prompt</button>
				<p></p>
				<button type="button" class = "Text2Prompt2" id="setLastText2Prompt">Get the Last Prompt</button>
				<p></p>
				<button type="button" class = "Text2Prompt" id="setRandomText2Prompt">Choose a Random Prompt</button>
				<p></P>
				<button type="button" class = "Text2Prompt" id="setText2Prompt">Use this Prompt</button>
				<p></p>
				<button title="Generate image(s)" class = "Text2Prompt" type="button" id="Text2PromptSingleRun">Generate Single Image</button>
				<p></p>
				<button title="Generate image(s)" class = "Text2Prompt" type="button" id="Text2PromptBatchRun">Generate Random Batch</button>
				<input title="Batch Quantity" type="number" id="Text2PromptBatchCount" value="4" min="1">
				<p></p>
				<button title="Generate all Wildcard images" type="button" class = "Text2Prompt" id="setText2SeqBatch">Generate Sequential Batch</button>
				<p></p>
			</div>
			`;
		BuildSettings.innerHTML = tempHTML;
		var editorSettings = document.getElementById('editor-settings');
		editorSettings.parentNode.insertBefore(BuildSettings, editorSettings.nextSibling);
		createCollapsibles(BuildSettings);

		document.getElementById ("importText2File").addEventListener ("click", getText2Prompt, false);
		document.getElementById ("setRandomText2Prompt").addEventListener ("click", setRandomText2Prompt, false);
		document.getElementById ("setText2Prompt").addEventListener ("click", setText2Prompt, false);
		document.getElementById ("text2PromptUseSearch").addEventListener ("click", text2PromptUseSearch, false);
		document.getElementById ("text2PromptClearSearch").addEventListener ("click", text2PromptClearSearch, false);
		document.getElementById ("Text2PromptSingleRun").addEventListener ("click", Text2PromptSingleRun, false);
		document.getElementById ("Text2PromptBatchRun").addEventListener ("click", Text2PromptBatchRun, false);
		document.getElementById ("setText2SeqBatch").addEventListener ("click", setText2SeqBatch, false);
		document.getElementById ("setFirstText2Prompt").addEventListener ("click", setFirstText2Prompt, false);
		document.getElementById ("setNextText2Prompt").addEventListener ("click", setNextText2Prompt, false);
		document.getElementById ("setPrevText2Prompt").addEventListener ("click", setPrevText2Prompt, false);
		document.getElementById ("setLastText2Prompt").addEventListener ("click", setLastText2Prompt, false);

		document.getElementById("text2promptCount").disabled = true;

		document.getElementById("text2prompt_Options2").addEventListener("keyup", e => {
			text2PromptInput = [...document.querySelectorAll('#WildcardList option')].map( option => option.value);
			validateTextInput(e.target.value)
		})

		document.querySelector('#text2prompt_Options2').classList.remove('valid');
		document.querySelector('#text2prompt_Options2').classList.add('invalid');

		document.getElementById ("setText2Prompt").style.display="none";

	}

	function setRandomText2Prompt() {
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				var randomText2Prompt = Math.floor(Math.random() * lenArray);
				document.getElementById ("text2prompt_Options").value = text2promptArray[randomText2Prompt];
				setText2Prompt();
				break;
		}

	}

	function setText2Prompt() {
		var preText2PromptField = document.getElementById ("preText2Prompt").value;
		var currText2PromptField = document.getElementById ("text2prompt_Options").value;
		var postText2PromptField = document.getElementById ("postText2Prompt").value;
		switch (currText2PromptField) {
			case null:
				break;
			case 'None':
				break;
			default:
				switch (preText2PromptField) {
					case null:
						break;
					case "":
						break;
					default:
						currText2PromptField = preText2PromptField + ',' + currText2PromptField;
						break;
				}
				switch (postText2PromptField) {
					case null:
						break;
					case "":
						break;
					default:
						currText2PromptField = currText2PromptField + ',' + postText2PromptField;
						break;
				}

				currText2Prompt.value = currText2PromptField;
				promptField.value = currText2Prompt.value;
				break;
		}
	}
	function Text2PromptSingleRun() {
		console.log('Create Single Image');
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		setText2Prompt();
		document.getElementById("makeImage").click();
	}

	function Text2PromptBatchRun() {
		console.log('Running Batch');
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		var batchCount = document.getElementById("Text2PromptBatchCount").value;
		for (let i = 0; i < batchCount; i++) {
			setRandomText2Prompt();
			setText2Prompt();
			document.getElementById("makeImage").click();
		}
	}

	function getText2Prompt() {
		console.log('Getting text from file');
		getText2PromptFile();
	}

	function getText2PromptFile() {
		var files = document.getElementById('Text2PromptSelectFiles').files;
		if (files.length <= 0) {
			return false;
		}
		clearMyText2PromptOptions();
		var mytext2promptCount = 1;
		var fr = new FileReader();
		fr.onload = function(e) {
			var myString = "";
			var myChar = null;
			//console.log('Text: ' + e.target.result);
			const inputs = Array.from(e.target.result);
			text2promptArray=[];
			inputs.forEach(input => {
				//console.log('Text Line: ' + input);
				myChar = input.charCodeAt(0);
				//console.log('Char: ' + input, myChar);
				switch (myChar) {
					case 10:
						console.log(myString);
						let position = myString.search("  ");
						//console.log('Position: ' + position);
						switch (position) {
							case -1:
								break;
							default:
								console.log('Replacing');
								let result = myString.replace("  ", " ");
								myString = result;
								break;
						}
						text2promptArray.push(myString);
						myString = "";
						mytext2promptCount = mytext2promptCount + 1;
						break;
					case 13:
						break;
					default:
						myString = myString + input;
						break;
				}
			});
			console.log(myString);
			text2promptArray.push(myString);
			populateMyText2PromptOptions();
			setFirstText2Prompt();
			document.getElementById("text2promptCount").value = mytext2promptCount;
		}
		fr.readAsText(files.item(0));
	}

	function populateMyText2PromptOptions() {
		text2promptArray.forEach((ItemFound) => {
			var x = document.getElementById("text2prompt_Options");
			var option = document.createElement("option");
			option.text = ItemFound;
			x.add(option);
			var myWildcardList = document.getElementById("WildcardList");
			var option = document.createElement("option");
			option.value = ItemFound;
			myWildcardList.appendChild(option);
		})
	}

	function clearMyText2PromptOptions() {
		text2promptArray = [];
		document.getElementById('text2prompt_Options').innerText = null;
		var x = document.getElementById('text2prompt_Options');
		var option = document.createElement("option");
		option.text = "None";
		x.add(option);
		document.getElementById('WildcardList').innerText = null;
	}


	function validateTextInput(myPassedVariable) {
		text2PromptInput = document.querySelectorAll('#WildcardList option');
		myCheckFlag = false;
		for (var i = 0; i < text2PromptInput.length; i++) {
			var obj = text2PromptInput[i].value;
			if ((obj).includes(myPassedVariable)) {
				myCheckFlag = true;
			}
		}
		switch (myCheckFlag) {
			case true:
				document.querySelector('#text2prompt_Options2').classList.remove('invalid');
				document.querySelector('#text2prompt_Options2').classList.add('valid');
				break;
			case false:
				document.querySelector('#text2prompt_Options2').classList.remove('valid');
				document.querySelector('#text2prompt_Options2').classList.add('invalid');
				break;
		}
	}

	function text2PromptUseSearch() {
		switch (myCheckFlag) {
			case true:
				document.getElementById ("text2prompt_Options").value = text2prompt_Options2.value;
				setText2Prompt();
				break;
			case false:
				break;
		}
	}

	function text2PromptClearSearch() {
		document.getElementById ("text2prompt_Options2").value = null;
		myCheckFlag = false;
		document.querySelector('#text2prompt_Options2').classList.remove('valid');
		document.querySelector('#text2prompt_Options2').classList.add('invalid');
	}

	function setText2SeqBatch() {
		console.log('Sequential Prompt');
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		var newText2Prompt = null;
		for (let arrayIndex = 0; arrayIndex < lenArray; arrayIndex++) {
				newText2Prompt = text2promptArray[arrayIndex];
				document.getElementById ("text2prompt_Options").value = newText2Prompt;
				setText2Prompt();
				document.getElementById("makeImage").click();
		}

	}

	function setFirstText2Prompt() {
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		console.log('Setting First Text2Prompt')
		var newText2Prompt = null;
		text2promptPosition = 0;
		newText2Prompt = text2promptArray[text2promptPosition];
		document.getElementById ("text2prompt_Options").value = newText2Prompt;
		setText2Prompt();
	}

	function setNextText2Prompt() {
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		console.log('Setting Next Text2Prompt')
		var newText2Prompt = null;
		text2promptPosition = text2promptPosition + 1;
		if (text2promptPosition > lenArray-1) {
			text2promptPosition = lenArray-1;
		}
		console.log('Text2Prompt Index: ' + text2promptPosition);
		newText2Prompt = text2promptArray[text2promptPosition];
		document.getElementById ("text2prompt_Options").value = newText2Prompt;
		setText2Prompt();
	}

	function setPrevText2Prompt() {
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		console.log('Setting Previous Text2Prompt')
		var newText2Prompt = null;
		text2promptPosition = text2promptPosition - 1;
		if (text2promptPosition < 0) {
			text2promptPosition = 0;
		}
		console.log('Text2Prompt Index: ' + text2promptPosition);
		newText2Prompt = text2promptArray[text2promptPosition];
		document.getElementById ("text2prompt_Options").value = newText2Prompt;
		setText2Prompt();
	}

	function setLastText2Prompt() {
		var lenArray = text2promptArray.length;
		switch (lenArray) {
			case 0:
				console.log('No Wildcard file loaded');
				return;
				break;
			default:
				break;
		}
		console.log('Setting Last Text2Prompt')
		var newText2Prompt = null;
		text2promptPosition = lenArray - 1;
		console.log('Text2Prompt Index: ' + text2promptPosition);
		newText2Prompt = text2promptArray[text2promptPosition];
		document.getElementById ("text2prompt_Options").value = newText2Prompt;
		setText2Prompt();
	}
})();
