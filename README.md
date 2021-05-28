### Setting Up

1. Download and install Python if not installed before.

   - Link: https://www.python.org/downloads/
   - Make shure that "Add to path" option is selected while installation.

2. Download repository and open the repository folder.

3. Open main.py with a text editor.

   - Open a new terminal and execute "ipconfig" command and copy the contents of ipv4 address. (Something like 192.168.1.144)
   - Paste the ip into host parameter in the last line of code.
   - e.g: uvicorn.run(app, host="192.168.1.144", port=8080)
   - Close this terminal.
   - Save and close the file.

4. Open a terminal in the repository folder.

5. Execute command: "pip install virtualenv" and wait for installation.

6. Execute command: "virtualenv venv" and wait for execution.

7. Execute command: "venv/Scripts/activate" and wait for execution.

8. Execute command: "pip install -r requirements.txt" and wait for installation.

9. Download our trained emotion detection model.

   - Link: https://drive.google.com/file/d/1XqUnAqVib9kYu3mnOENgjt6GUCKvnpdG/view?usp=sharing
   - Move the emotion.h5 file into assets folder of project.

10. Execute command: "python -m main.py" and wait for execution.
