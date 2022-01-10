from typing import Dict, List

from classifier.classifier import Classifier
from config import settings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic.main import BaseModel
from sqlalchemy.orm import Session

from . import models
from .database import SessionLocal, engine
from .models import Images

# Create db & table with SQLalchemy
models.Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic classes
class ImageRequest(BaseModel):
    image: str


# class ImageResponse(BaseModel):
#     probabilities: Dict[str, float]
#     sentiment: str
#     confidence: float


# FastAPI
app = FastAPI(
    title=settings.name, description=settings.description, version=settings.version
)

# Jinja2Templates
templates = Jinja2Templates(
    directory="D:/CompSci/Projects/digit-drawing-prediction/digit_recognition/templates/"
)

# NLP classification background_task
def predict_image(
    id: int,
):
    """
    -------
    """
    db = SessionLocal()
    image = db.query(Images).filter(Images.id == id).first()
    prediction_output = Classifier(image.image).return_list()
    image.output = prediction_output
    db.add(image)
    db.commit()


# Routes
@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    -------
    """
    images = db.query(Images).all()
    return templates.TemplateResponse(
        "layout.html",
        {
            "request": request,
            "images": images,
        },
    )


@app.post("/image")
def create_image(
    image_request: ImageRequest,
    db: Session = Depends(get_db),
):
    """
    background_tasks: BackgroundTasks,
    background_tasks.add_task(predict_image, image.id)
    (2) add database record
    (3) give background_tasks a reference of image record
    """
    image = Images()
    image.image = image_request.image
    # image.output = "4"  # <<< image_request.output
    db.add(image)
    db.commit()

    return {
        "code": "success",
        "message": "image added",
    }
