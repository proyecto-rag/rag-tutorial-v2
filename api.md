# FLASK BASICS
## Inside env
To run your web application, you’ll first tell Flask where to find the application (api.py) with the FLASK_APP environment variable:

```bash
export FLASK_APP=api
```

Then run it in development mode with the FLASK_ENV environment variable:
```bash
export FLASK_ENV=development
```

Run api
```bash
flask run
```