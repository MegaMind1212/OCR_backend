import serverless_wsgi
from backend import app

# Vercel serverless function handler
def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)