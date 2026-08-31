/*
    Toggle Spellcheck
    by Patrice

    Adds a system setting to turn the spell check on and off for the prompt, negative prompt, and image modifiers text areas.
*/
(function() {
    "use strict"

    /* inject new settings in the existing system settings popup table */
    let settings = [
        {
            id: "enable_spellcheck",
            type: ParameterType.checkbox,
            label: "Enable spellcheck",
            note: "Enable spell check for prompt, negative prompt, and image modifiers",
            icon: "fa-spell-check",
            default: false
        }
    ];

    PARAMETERS.push(...settings)
    prettifyInputs(document);
    let enableSpellcheck = document.querySelector("#enable_spellcheck")

    // save/restore the setting
    enableSpellcheck.addEventListener('change', (e) => {
        localStorage.setItem(settings[0].id, enableSpellcheck.checked)
        updateSpellcheck()
    })
    enableSpellcheck.checked = localStorage.getItem(settings[0].id) == null ? settings[0].default : localStorage.getItem(settings[0].id) === 'true'
    updateSpellcheck()

    function updateSpellcheck() {
        promptField.setAttribute("spellcheck", enableSpellcheck.checked ? "true" : "false")
        negativePromptField.setAttribute("spellcheck", enableSpellcheck.checked ? "true" : "false")
        customModifiersTextBox.setAttribute("spellcheck", enableSpellcheck.checked ? "true" : "false")
    }
})()
