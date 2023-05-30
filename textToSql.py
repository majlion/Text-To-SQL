from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the pre-trained T5 model and tokenizer
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the API endpoint for text-to-SQL conversion
@app.route('/convert', methods=['POST'])
def convert_text_to_sql():
    text = request.json['text']  # Get the input text from the request JSON

    # Convert text to SQL query
    sql_query = convert_to_sql(text)

    return jsonify({'sql_query': sql_query})  # Return the SQL query as JSON response

def convert_to_sql(text):
    # Tokenize the text and generate the SQL query
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return sql_query

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask app
