#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies and build
cd frontend
npm install
npm run build
cd ..
