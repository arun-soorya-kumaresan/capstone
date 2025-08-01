# run.py
from app import app

# --- Run the App ---
if __name__ == '__main__':
    # Setting debug=True allows you to see errors and automatically reloads the app when you change code.
    # Turn this to False for production.
    app.run(debug=True)