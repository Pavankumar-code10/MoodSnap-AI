# MoodSnap AI

Smart Story Photo Selector powered by AI.

## Overview

MoodSnap AI helps users quickly choose the best photo for social media stories or posts based on their current mood.

The user enters a mood prompt such as:

- happy but confused
- missing old memories
- powerful energy
- peaceful evening vibes

The AI scans images from the local gallery folder and suggests the most matching photos.

---

## Problem Statement

Many users spend time selecting photos for Instagram stories, WhatsApp status, or social media posts.

Choosing the right image that matches emotion or mood can be difficult and time-consuming.

---

## Solution

MoodSnap AI uses a multimodal AI model (CLIP) that understands both:

- Text meaning
- Image meaning

It compares the user’s mood prompt with available images and ranks the best matches.

---

## Features

- Mood-based photo recommendation
- Best matching image popup
- Top 5 ranked image suggestions
- Fast local image scanning
- Easy to use Python application

---

## Tech Stack

- Python
- PyTorch
- CLIP Model
- Pillow (PIL)

---

## Project Structure

```text
MoodSnap-AI/
│── app.py
│── requirements.txt
│── README.md
│── images/
