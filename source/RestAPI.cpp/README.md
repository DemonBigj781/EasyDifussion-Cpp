# RestAPI.cpp

Migration target for the network-facing REST and cURL portion of the current
SDKIT3 project.

RestAPI.cpp owns transport concerns such as HTTP requests, endpoints,
serialization, authentication boundaries, and cURL integration. It remains
separate from UI.cpp, INFERENCE.cpp, and DiffUser.cpp. Together, RestAPI.cpp,
UI.cpp, and INFERENCE.cpp are the three destinations for breaking up SDKIT3.
Their dependency contracts will be defined before source migration begins.
