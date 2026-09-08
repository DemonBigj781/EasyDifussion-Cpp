# UI.cpp

Migration target for the UI-facing portion of the current SDKIT project.

UI.cpp will own presentation, user interaction, and UI-facing orchestration. It
must not become an inference implementation or a hardware-driver abstraction.
This split does not change DiffUser.cpp or existing supporting components. Its
contracts with the other project roots will be documented before source
migration begins.
