# PIDPlus

A measurement model based on a CNN !

## How to use it

### 1 Basic requirements
This project is pretty self contained, but there is some basic dependencies for running it. You will need
- Linux File System 
- Docker 
- Duckietown Shell
### 2. Executable files
Make sure that our file system can execute our nodes as executable files. This can be done using the command 
```
chmod +x packages/pidplus/src/model_node.py packages/pidplus/src/main_control_node.py
```
### 3. Build the project
For building the project, we use the following command
```
dts devel build -f
```
This might take some time as it needs to pull our docker image containing pytorch. Since we are building the project locally to run on our computer, we do not reference a robot name.

### 4   . Run the project
For running the project, we use the following command
```
dts devel run -R ROBOT_NAME -L full-stack -X
```
Note that the full-stack launcher launches the `main-control` node and the `model` node at the same time, on your local machine since it is where it built. The `-X` is optional, but proved to be useful in debugging for allowing the project to create a new window and the images the duckiebot saw.
