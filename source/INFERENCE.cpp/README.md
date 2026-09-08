# INFERENCE.cpp

Migration target for the inference portion of the current SDKIT project.

INFERENCE.cpp will own inference orchestration and implementations developed for
this project. This split does not change DiffUser.cpp or existing supporting
components. SDKIT3 and llama.cpp remain research references rather than runtime
dependencies or code donors after migration. Contracts between project roots
will be documented before source migration begins.
