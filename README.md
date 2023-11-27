# NATproxy_browser 
Welcome to NATproxy!

Hello and welcome! This project is a proxy server built using Flask and aims to provide a secure browsing experience. Below are the instructions to get started:

## Prerequisites

- Python installed on your system
- Command-line interface (CLI) such as Terminal, Command Prompt, or PowerShell
- Access to the project's source code

## Installation

1. **Upgrade or Install pip**: Open a command prompt and execute:
    
    py -m pip install --upgrade pip
    

2. **Install project dependencies**: Navigate to the project directory and run:
   
    pip install -r requirements.txt
    
    If this doesn't work, install each library separately required libraries are in requirements.txt :
    
    pip install "library-name"

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Usage

1. **Locate the app.py file**: 
Within the project source code, find the `app.py` file.

2. **Run the application**:
 Execute the `app.py` file. You should receive a link to the local host where the server is running in the console.

3. **Login instructions**:

    - Use the following credentials for user login:
        - Email: at56@mail.aub.edu
        - Password: 12345
        (This email is added specifically for testing purposes.)
	after logging in browse the on the proxy freely

    - To log in as an admin:
        - Email: natproxyy@outlook.com()
        - Password: NATPROXY123
        (These credentials grant access to the admin panel where you can add/delete users and Block/Unblock URLS)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Logging

All logs related to the precise time and date of the requests, as well as information about the ML tool, will be logged both to the console and saved into a `logs.txt` file.

### Logging Behavior

- **Console Output**: Logs related to request times, dates, and ML tool details will be printed to the console during application runtime.
  
- **File Storage**: Simultaneously, these logs will be saved to a file named `logs.txt`.

### Potential Log Duplication

Note: There might be instances where logs get duplicated in the `logs.txt` file, depending on the Python version being used. Efforts have been made to mitigate this issue within the code; however, duplicate log entries might occur in certain Python environments.
