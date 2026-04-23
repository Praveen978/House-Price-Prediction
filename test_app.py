from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1 style="color: green; text-align: center; margin-top: 100px;">
            HELLO LEKHA! THIS IS A TEST PAGE — IT WORKS! 🎉
        </h1>
        <p style="text-align: center; font-size: 20px;">
            If you see this, Flask + templates are working correctly.
        </p>
    </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(debug=True, port=5000)