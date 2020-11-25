from flask import Flask, request
from app import main

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=["GET", "POST"])
def adder_page():
    errors = ""
    if request.method == "POST":
        text = list(request.form["text"])

        if text is not None:
            result = main(text)
            if result[0] == 1:
                result = "Positive"
            else:
                result = "Negative"
            return '''
                <html>
                    <body>
                        <p>The result is {result}</p>
                        <p><a href="/">Click here to predict again</a>
                    </body>
                </html>
            '''.format(result=result)

    return '''
        <html>
            <body>
                {errors}
                <p>Enter your text:</p>
                <form method="post" action=".">
                    <p><input name="text" /></p>
                    <p><input type="submit" value="Predict Sentiment" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)


if __name__ == '__main__':
    app.run(debug=True)
