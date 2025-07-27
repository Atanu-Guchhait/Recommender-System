from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create FastAPI app
app = FastAPI()

# Static files and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load CSV files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Database setup
DATABASE_URL = "mysql+pymysql://root:atanukuchu%40123@localhost/ecom"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Signup(Base):
    __tablename__ = "signup"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    password = Column(String(100), nullable=False)


Base.metadata.create_all(bind=engine)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n + 1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

random_image_urls = [
    "static/img/img_1.png", "static/img/img_2.png", "static/img/img_3.png",
    "static/img/img_4.png", "static/img/img_5.png", "static/img/img_6.png",
    "static/img/img_7.png", "static/img/img_8.png",
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "trending_products": trending_products.head(8),
            "truncate": truncate,
            "random_product_image_urls": random_product_image_urls,
            "random_price": random.choice(price),
        },
    )

@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])
    return templates.TemplateResponse("main.html", {"request": request, "content_based_rec": content_based_rec})

@app.post("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request, prod: str = Form(...), nbr: str = Form(...)):
    if not nbr.isdigit():
        message = "Please enter a valid number of recommendations."
        empty_df = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])
        return templates.TemplateResponse(
            "main.html",
            {"request": request, "message": message, "content_based_rec": empty_df}
        )
    nbr = int(nbr)
    content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
    if content_based_rec.empty:
        message = "No recommendations available for this product."
        empty_df = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating'])
        return templates.TemplateResponse(
            "main.html",
            {"request": request, "message": message, "content_based_rec": empty_df}
        )
    else:
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return templates.TemplateResponse(
            "main.html",
            {
                "request": request,
                "content_based_rec": content_based_rec,
                "truncate": truncate,
                "random_product_image_urls": random_product_image_urls,
                "random_price": random.choice(price),
            },
        )

# Signup route
@app.post("/signup")
async def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing_user = db.query(Signup).filter(or_(Signup.username == username, Signup.email == email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists.")
    new_user = Signup(username=username, email=email, password=password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Signup successful", "user": new_user.username}

# Signin route
@app.post("/signin")
async def signin(signinUsername: str = Form(...), signinPassword: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(Signup).filter(
        or_(Signup.username == signinUsername, Signup.email == signinUsername),
        Signup.password == signinPassword
    ).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username/email or password")
    return {"message": f"Welcome back, {user.username}!"}
