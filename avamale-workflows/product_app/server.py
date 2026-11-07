"""Entrypoint with secure-by-default local bind and mandatory admin credentials."""
from __future__ import annotations
import os
import uvicorn
from .api import create_app

app = create_app()  # fails closed until environment secrets are configured

if __name__ == '__main__':
    port = int(os.getenv('PORT', '8000'))
    host = '0.0.0.0' if os.getenv('AVAMALE_PUBLIC') == '1' else '127.0.0.1'
    uvicorn.run('product_app.server:app', host=host, port=port, access_log=False)
