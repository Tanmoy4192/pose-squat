## How to Run the Program

1. Open the project folder in the terminal.

2. Create a virtual environment.

```
python -m venv venv
```

3. Activate the virtual environment.

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

4. Install the required libraries.

```
pip install -r requirements.txt
```

5. Run the FastAPI server.

```
uvicorn app.main:app --reload
```

6. Open the browser and go to:

```
http://127.0.0.1:8000/video
```

The webcam stream will start and the squat analyzer will run.
