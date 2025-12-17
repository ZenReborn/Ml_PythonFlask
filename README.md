How to run

1. Create a virtualenv and install:

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt


2. Train the model:

python train.py


3. This produces saved_model/ with the trained model and fit_plot.png.

4. Run the Flask app:

python app.py

5. Finally open http://127.0.0.1:5000/ in web browsr
Done
