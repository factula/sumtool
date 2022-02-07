# sumtool

## References

[References](reference.md)

## Setup
Setup (python 3.8):
```
pip install -r requirements.txt
```

### Run Streamlit app
```
streamlit run interface/app.py
```

You can also run interfaces individually, e.g.
```
streamlit run interface/factuality_interface.py
```

### Contributors

Setup (python 3.8):
```
pip install -r requirements.dev.txt
```

Before commiting:
```
black sumtool/
flake8 sumtool/
```

