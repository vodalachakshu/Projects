# Desk Agent

A simple LangChain desk agent that understands natural-language requests and routes them to two predefined tools:

- `lookup_customer` — looks up a customer by ID (`C100` or `C200`)
- `check_service_status` — checks `billing`, `authentication`, or `support`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env
# Add your OpenAI API key to .env
