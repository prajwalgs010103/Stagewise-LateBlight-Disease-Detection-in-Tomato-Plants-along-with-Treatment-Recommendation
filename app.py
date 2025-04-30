import os
import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash
import datetime
import base64
from flask import Flask, request, render_template, session, redirect, url_for
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from flask import Flask, render_template, request, send_file, session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.lib.utils import ImageReader
import json
import os
import datetime
import base64
import torch
import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tensorflow.keras.preprocessing import image
import gdown
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'KibutzujiMuzan@1234'  # Replace with a secret key for session management
cred = credentials.Certificate("stagewisedetection-firebase-adminsdk-fbsvc-9bb646ffdd.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# File upload configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home Route (Accessible only after login)
@app.route('/')
def home():
    # if 'username' not in session:
    #     return redirect(url_for('login'))  # Redirect to login page if not logged in
    return render_template('home.html')

# Load the trained model
# model_path = "inceptionv3_model6.h5"
# model = tf.keras.models.load_model(model_path)

# # Class labels
# class_labels = ["Healthy","Late Blight Stage 1", "Late Blight Stage 2", "Late Blight Stage 3"]

import gdown
import tensorflow as tf

# Google Drive file ID
file_id = "1giyFsfzb-pwtR7cerkfoCkpKNRvLHuc1"
model_path = "inceptionv3_model6.h5"

# Download model if it doesn't exist locally
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)


# Load model
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ["Healthy", "Late Blight Stage 1", "Late Blight Stage 2", "Late Blight Stage 3"]


UPLOAD_FOLDER = 'static/uploads/'  # Local folder to temporarily store images
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#Single image detection for stage
# @app.route('/detect', methods=['GET', 'POST'])
# def detect():
#     if 'username' not in session:
#         return redirect(url_for('login'))

#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return "No file part", 400
#         file = request.files['file']

#         if file.filename == '':
#             return "No selected file", 400

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)

#             # Save original image
#             file.save(filepath)

#             # Load image for model prediction (299x299)
#             img = image.load_img(filepath, target_size=(299, 299))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0) / 255.0

#             # Predict disease stage
#             predictions = model.predict(img_array)
#             predicted_class = np.argmax(predictions)
#             disease_stage = class_labels[predicted_class]

#             # Reduce size **only for Firestore storage**
#             img = Image.open(filepath)
#             img = img.resize((400, 400), Image.LANCZOS)  # ✅ NEW (Works in latest Pillow)
#             compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
#             img.save(compressed_path, quality=85)  # Save with 85% quality

#             # Convert **compressed** image to Base64 for Firestore
#             with open(compressed_path, "rb") as image_file:
#                 img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

#             # Get current date and time
#             timestamp = datetime.datetime.now().isoformat()

#             # Save result to Firestore
#             user_id = session['username']
#             doc_ref = db.collection('users').document(user_id).collection('detections').add({
#                 'image_base64': img_base64,
#                 'filename': filename,
#                 'disease_stage': disease_stage,
#                 'timestamp': timestamp
#             })

#             return render_template('result.html', disease_stage=disease_stage, image_file=filename, timestamp=timestamp)

#     return render_template('detect.html')


from huggingface_hub import hf_hub_download
repo_id = "foduucom/plant-leaf-detection-and-classification"
filename = "best.pt"

model_path4 = hf_hub_download(repo_id=repo_id, filename=filename)
yolo_model = YOLO(model_path4)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # ✅ Save original image
            file.save(filepath)

            # ✅ Step 1: Run YOLO Model for Object Detection
            results = yolo_model(filepath)
            detected_labels = []

            for result in results:
                for box in result.boxes.cls:
                    label = result.names[int(box.item())]  # Get detected object name
                    detected_labels.append(label)

            print(f"🔍 Detected Objects: {detected_labels}")

            # ✅ Check if a vegetable or leaf is detected
            vegetable_keywords = ["leaf", "vegetable", "tomato", "plant", "potato"]
            is_vegetable = any(any(keyword in label.lower() for keyword in vegetable_keywords) for label in detected_labels)

            # ✅ Resize and Convert Image for Firestore Storage
            img = Image.open(filepath)
            img = img.resize((400, 400), Image.LANCZOS)
            compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
            img.save(compressed_path, quality=85)

            with open(compressed_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            if is_vegetable:
                # ✅ Step 2: Run Inception V3 Model for Disease Classification
                img_array = image.img_to_array(image.load_img(filepath, target_size=(299, 299)))
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                disease_stage = class_labels[predicted_class]
            else:
                disease_stage = "No leaf Detected"

            # ✅ Store result in Firestore (Always Store the Image)
            timestamp = datetime.datetime.now().isoformat()
            user_id = session['username']
            db.collection('users').document(user_id).collection('detections').add({
                'image_base64': img_base64,
                'filename': filename,
                'disease_stage': disease_stage,
                'detected_objects': detected_labels,
                'timestamp': timestamp
            })

            return render_template('result.html', disease_stage=disease_stage, image_file=filename, 
                                   detected_objects=detected_labels, timestamp=timestamp)

    return render_template('detect.html')






# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

#Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("Form Data:", request.form)  # Debugging line

        username = request.form.get('username')  # Get the username
        password = request.form.get('password')  # Get the password

        if not username or not password:
            return "Username or password missing", 400  # Return error if fields are missing

        try:
            # Search Firestore for user by username
            users_ref = db.collection('users').where('username', '==', username).stream()
            user_doc = None
            for u in users_ref:
                user_doc = u
                break  # Get the first matching user

            if user_doc:
                user_data = user_doc.to_dict()
                stored_password = user_data.get('password')

                # Compare passwords (assuming plaintext, but should be hashed)
                if check_password_hash(user_data['password'], password):
                    session['username'] = username
                    #flash('Login successful!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid password', 'danger')
                    return "Invalid password", 403
            else:
                flash('Username not found', 'danger')
                return "Username not found", 403

        except Exception as e:
            return f"Error: {str(e)}", 403

    return render_template('login.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        dob = request.form['dob']
        contact = request.form['contact']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "error")
            return redirect(url_for('register'))

        try:
            # Check if email already exists
            auth.get_user_by_email(email)
            flash("Email is already registered. Try logging in.", "error")
            return redirect(url_for('register'))
        except firebase_admin.auth.UserNotFoundError:
            pass  # Email does not exist, proceed with registration

        try:
            # Create new user in Firebase Authentication
            user = auth.create_user(email=email, password=password)

            # Hash the password before storing it
            hashed_password = generate_password_hash(password)

            # Save user details to Firestore
            db.collection('users').document(user.uid).set({
                'first_name': first_name,
                'last_name': last_name,
                'dob': dob,
                'contact': contact,
                'username': username,
                'email': email,
                'password': hashed_password  # Store hashed password
            })

            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('register'))

    return render_template('register.html')

#Single image stage detection history
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = session['username']
    limit = request.args.get('limit', default=10, type=int)  # Default to 10 records

    # Fetch predictions from Firestore, sorted by timestamp (latest first)
    detections = (
        db.collection('users')
        .document(user_id)
        .collection('detections')
        .order_by('timestamp', direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )

    history = [
        {
            "image_base64": entry.to_dict().get("image_base64"),
            "disease_stage": entry.to_dict().get("disease_stage"),
            "timestamp": entry.to_dict().get("timestamp"),
        }
        for entry in detections
    ]

    return render_template("history.html", history=history, selected_limit=limit)

@app.route('/detect_multiple', methods=['GET', 'POST'])
def detect_multiple():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'files' not in request.files:
            return "No file part", 400
        files = request.files.getlist('files')

        if not files or all(file.filename == '' for file in files):
            return "No selected files", 400

        user_id = session['username']
        timestamp = datetime.datetime.now().isoformat()
        predictions_list = []
        class_counts = {}

        # Define disease class ranking
        class_rank = {
            "Healthy": 0,
            "Late Blight Stage 1": 1,
            "Late Blight Stage 2": 2,
            "Late Blight Stage 3": 3
        }

        detected_classes = set()  # Store detected disease classes

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                # ✅ Save original image
                file.save(filepath)

                # ✅ Step 1: Run YOLO Model for Object Detection
                results = yolo_model(filepath)
                detected_labels = []

                for result in results:
                    for box in result.boxes.cls:
                        label = result.names[int(box.item())]  # Get detected object name
                        detected_labels.append(label)

                print(f"🔍 Detected Objects in {filename}: {detected_labels}")

                # ✅ Check if a vegetable or leaf is detected
                vegetable_keywords = ["leaf", "vegetable", "tomato", "plant", "potato"]
                is_vegetable = any(any(keyword in label.lower() for keyword in vegetable_keywords) for label in detected_labels)

                if is_vegetable:
                    # ✅ Step 2: Run Inception V3 Model for Disease Classification
                    img = image.load_img(filepath, target_size=(299, 299))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions)
                    disease_stage = class_labels[predicted_class]

                    # Track detected disease class counts
                    class_counts[disease_stage] = class_counts.get(disease_stage, 0) + 1
                    detected_classes.add(disease_stage)
                else:
                    disease_stage = "No leaf Detected"

                # ✅ Compress image for Firestore storage
                img = Image.open(filepath)
                img = img.resize((400, 400), Image.LANCZOS)
                compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
                img.save(compressed_path, quality=85)

                # ✅ Convert compressed image to Base64
                with open(compressed_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # ✅ Store prediction result
                predictions_list.append({
                    'image_base64': img_base64,
                    'filename': filename,
                    'disease_stage': disease_stage,
                    'detected_objects': detected_labels,
                    'timestamp': timestamp
                })

        # ✅ Determine the highest detected disease class (if any vegetable was detected)
        highest_class = max(detected_classes, key=lambda x: class_rank[x]) if detected_classes else "No Disease Detected"

        # ✅ Store results in Firestore
        db.collection('users').document(user_id).collection('batch_detections').add({
            'timestamp': timestamp,
            'highest_class': highest_class,
            'predictions': predictions_list,
            'class_counts': class_counts
        })

        return render_template('multiple_result.html', predictions=predictions_list, 
                               timestamp=timestamp, highest_class=highest_class, 
                               class_counts=class_counts)

    return render_template('detect_multiple.html')








#History of mutiple images stage
@app.route('/history_multiple')
def history_multiple():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = session['username']
    limit = request.args.get('limit', default=10, type=int)

    # Fetch the latest `limit` documents from Firestore
    docs = db.collection('users').document(user_id).collection('batch_detections')\
             .order_by('timestamp', direction=firestore.Query.DESCENDING)\
             .limit(limit).stream()

    history = []
    for doc in docs:
        data = doc.to_dict()
        history.append({
            'timestamp': data['timestamp'],
            'predictions': data['predictions'],
            'highest_class': data['highest_class']  # Fetch stored highest class
        })

    return render_template('history_multiple.html', history=history, selected_limit=limit)

#Disease Detection
processor = AutoImageProcessor.from_pretrained("Prajwal-113/TomatoDisease")
image_classification_model = AutoModelForImageClassification.from_pretrained("Prajwal-113/TomatoDisease")



@app.route('/new_detect', methods=['GET', 'POST'])
def new_detect():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'files' not in request.files:
            return "No file part", 400
        files = request.files.getlist('files')

        if not files or all(file.filename == '' for file in files):
            return "No selected files", 400

        user_id = session['username']
        timestamp = datetime.datetime.now().isoformat()  # Common timestamp for all images
        predictions_list2 = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                # ✅ Save original image
                file.save(filepath)

                # ✅ Step 1: Run YOLO Model for Object Detection
                results = yolo_model(filepath)
                detected_labels = []

                for result in results:
                    for box in result.boxes.cls:
                        label = result.names[int(box.item())]  # Get detected object name
                        detected_labels.append(label)

                print(f"🔍 Detected Objects in {filename}: {detected_labels}")

                # ✅ Check if a vegetable or leaf is detected
                vegetable_keywords = ["leaf", "vegetable", "tomato", "plant", "potato"]
                is_vegetable = any(any(keyword in label.lower() for keyword in vegetable_keywords) for label in detected_labels)

                if is_vegetable:
                    # ✅ Step 2: Run Image Classification Model for Disease Prediction
                    img = image.load_img(filepath, target_size=(299, 299))
                    inputs = processor(images=img, return_tensors="pt")

                    with torch.no_grad():
                        outputs = image_classification_model(**inputs)
                        logits = outputs.logits
                        predicted_class_idx = logits.argmax(-1).item()
                        predicted_label = image_classification_model.config.id2label[predicted_class_idx]
                else:
                    predicted_label = "No leaf Detected"

                # ✅ Compress image before storing
                img = Image.open(filepath)
                img = img.resize((400, 400), Image.LANCZOS)
                compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
                img.save(compressed_path, quality=85)

                # ✅ Convert compressed image to Base64
                with open(compressed_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # ✅ Append prediction result
                predictions_list2.append({
                    'image_base64': img_base64,
                    'filename': filename,
                    'disease_stage': predicted_label,
                    'detected_objects': detected_labels,
                    'timestamp': timestamp
                })

        # ✅ Store all images under the same timestamp in Firestore
        db.collection('users').document(user_id).collection('new_detections').add({
            'timestamp': timestamp,
            'predictions': predictions_list2
        })

        return render_template('new_detect_result.html', predictions=predictions_list2, timestamp=timestamp)

    return render_template('new_detect.html')







#Profile page
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']  # Get logged-in username
    
    # Fetch user details based on username (assuming 'username' is a field in Firestore)
    users_ref = db.collection('users')
    query = users_ref.where('username', '==', username).limit(1).get()

    if not query:
        return "User not found", 404

    user_doc = query[0]
    user_info = user_doc.to_dict()

    if request.method == 'POST':
        updated_data = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'dob': request.form['dob'],
            'contact': request.form['contact']
        }

        user_doc.reference.update(updated_data)  # Update Firestore
        return redirect(url_for('profile'))  # Refresh page after update

    return render_template('profile.html', user=user_info)

#Forgot Password
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        
        # Check if username and email exist in Firebase
        users_ref = db.collection('users')
        query = users_ref.where('username', '==', username).where('email', '==', email).stream()
        user_doc = next(query, None)

        if user_doc:
            # Show new password fields
            return render_template('forgot_password.html', show_reset=True, username=username, email=email)
        else:
            flash("User not found. Please check your details.", "danger")

    return render_template('forgot_password.html', show_reset=False)

#Reset Password
@app.route('/reset_password', methods=['POST'])
def reset_password():
    username = request.form['username']
    email = request.form['email']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']

    if new_password != confirm_password:
        flash("Passwords do not match!", "danger")
        return redirect(url_for('forgot_password'))

    # Hash the new password
    hashed_password = generate_password_hash(new_password)

    # Find user and update password
    users_ref = db.collection('users')
    query = users_ref.where('username', '==', username).where('email', '==', email).stream()
    user_doc = next(query, None)

    if user_doc:
        user_id = user_doc.id  # Get Firestore document ID
        db.collection('users').document(user_id).update({'password': hashed_password})
        flash("Password updated successfully!", "success")
        return redirect(url_for('login'))
    else:
        flash("User not found!", "danger")
        return redirect(url_for('forgot_password'))

#Detected Disease History
@app.route('/disease_history')
def disease_history():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = session['username']
    
    # Get limit from the request (default to 10 if not provided)
    limit = request.args.get('limit', default=10, type=int)
    
    # Fetch the latest `limit` documents from Firestore
    docs = db.collection('users').document(user_id).collection('new_detections')\
             .order_by('timestamp', direction=firestore.Query.DESCENDING)\
             .limit(limit).stream()

    history = []
    for doc in docs:
        data = doc.to_dict()
        history.append({
            'timestamp': data['timestamp'],
            'predictions': data['predictions']
        })
    return render_template('disease_history.html', history=history, selected_limit=limit)

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

#Single Image detection report
@app.route('/single_report')
def single_report():
    user_id = session.get("username", "Guest")
    disease_stage = request.args.get("disease_stage")
    timestamp = request.args.get("timestamp")
    image_file = request.args.get("image_file")

    # **Check if Image File Exists**
    image_path = os.path.join(UPLOAD_FOLDER, image_file)
    if not os.path.exists(image_path):
        return "Error: Image file not found", 404

    # **Create PDF**
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # **Title Section**
    title_height = height - 80
    c.setFillColorRGB(0.85, 0.85, 0.85)
    c.rect(30, title_height, width - 60, 40, fill=1)
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0, 0, 0)
    c.drawCentredString(width / 2, title_height + 15, "Tomato Leaf Disease Detection Report")

    # **Detection Report Subheading**
    report_height = title_height - 30
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, report_height, "Detection Report")

    # **User and Disease Details**
    details_x, details_y = 50, report_height - 30  # Move the box slightly up
    box_width, box_height = 530, 100  # Increased height by 10 to raise the upper line
    c.rect(details_x - 10, details_y - box_height, box_width, box_height)


    c.setFont("Helvetica", 12)
    details_text = [
        f"User: {user_id}",
        f"Detection Date: {timestamp.split('T')[0]}",
        f"Detection Time: {timestamp.split('T')[1].split('.')[0]}",
        f"Detected Disease Stage: {disease_stage}"
    ]
    
    for i, text in enumerate(details_text):
        c.drawString(details_x, details_y - (i * 20)-25, text)

    # **Uploaded Image Label**
    label_y = details_y - box_height - 20
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, label_y, "Uploaded Image")

    # **Insert Image (Centered and Lowered)**
    try:
        img_reader = ImageReader(image_path)
        img_width, img_height = img_reader.getSize()
        aspect_ratio = img_width / img_height
        new_width = 250
        new_height = new_width / aspect_ratio

        x_position = (width - new_width) / 2
        y_position = label_y - new_height - 20  # Lowered for better alignment

        c.drawImage(img_reader, x_position, y_position, width=new_width, height=new_height)
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

    # Footer text
    footer_text = "© 2025 Tomato Leaf Disease Detector. All Rights Reserved."

    # Get page width and set Y position for footer
    footer_y = 30  # Position from bottom
    c.setFont("Helvetica-Oblique", 10)

    # Draw the footer text centered
    c.drawCentredString(width / 2, footer_y, footer_text)


    # **Separator Lines**
    c.line(30, title_height - 5, width - 30, title_height - 5)
    c.line(30, report_height - 17, width - 30, report_height - 17)
    c.line(30, details_y - box_height - 5, width - 30, details_y - box_height - 5)

    # **Save and Return PDF**
    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="detection_report.pdf", mimetype="application/pdf")

#Multiple images Stage Detection report
@app.route('/mul_img_stage_report')
def mul_img_stage_report():
    user_id = session.get("username", "Guest")
    timestamp = request.args.get("timestamp")

    # Fetch the batch data from Firestore
    batch_docs = db.collection('users').document(user_id).collection('batch_detections')\
        .where("timestamp", "==", timestamp).stream()

    batch_data = None
    for doc in batch_docs:
        batch_data = doc.to_dict()
        break  # We only need the first matching document

    if not batch_data:
        return "Error: No matching record found", 404

    predictions = batch_data.get("predictions", [])
    highest_class = batch_data.get("highest_class", "N/A")
    class_counts = batch_data.get("class_counts", {})

    # **Create PDF**
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # **Title Section**
    title_height = height - 80
    c.setFillColorRGB(0.22, 0.56, 0.24)  # Green background
    c.rect(30, title_height, width - 60, 40, fill=1)
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(1, 1, 1)  # White text
    c.drawCentredString(width / 2, title_height + 15, "Tomato Leaf Disease Detection Report")

    # **Detection Report Subheading**
    report_height = title_height - 30
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0)  # Black text
    c.drawCentredString(width / 2, report_height, "Detection Report")

    # **User and Disease Details Box**
    details_x, details_y = 50, report_height - 30
    box_width, box_height = 450, 110
    c.rect(details_x - 10, details_y - box_height, box_width, box_height)

    c.setFont("Helvetica", 12)
    details_text = [
        f"User: {user_id}",
        f"Date: {timestamp.split('T')[0]}",
        f"Time: {timestamp.split('T')[1].split('.')[0]}",
        f"Highest Detected Stage: {highest_class}"
    ]

    for i, text in enumerate(details_text):
        c.drawString(details_x, details_y - (i * 20) - 25, text)

    # **Move only the first row of images down**
    img_y = details_y - box_height - 120  # First row forced lower

    # **Uploaded Images & Stages**
    img_x, img_width, img_height = 50, 150, 100
    img_spacing = 30
    images_per_row = 3
    first_row = True  # Flag to track if it's the first row

    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, details_y - box_height - 80, "Uploaded Images")

    img_y -= 30  # Adjust space after heading

    for i, prediction in enumerate(predictions):
        try:
            # Convert base64 image back to binary
            image_data = base64.b64decode(prediction["image_base64"])
            img_reader = ImageReader(BytesIO(image_data))

            # Calculate positions for multiple rows
            x_pos = img_x + (i % images_per_row) * (img_width + img_spacing)

            # **Only move the first row further down**
            if first_row and i < images_per_row:
                img_y -= 50  # Push first row down more
                first_row = False  # Reset flag after first row is placed

            # **Check for space on the page, if not start a new page**
            if img_y - img_height < 100:
                c.showPage()
                img_y = height - 100  # Reset position on new page
                c.setFont("Helvetica-Bold", 12)
                img_y -= 40  # Adjust space after heading

            c.drawImage(img_reader, x_pos, img_y, width=img_width, height=img_height)

            # Display disease stage below image
            c.setFont("Helvetica", 11)
            c.drawCentredString(x_pos + img_width / 2, img_y - 30, f"Stage: {prediction['disease_stage']}")

            # Move to the next row if necessary
            if (i + 1) % images_per_row == 0:
                img_y -= img_height + 40  # Normal row spacing

        except Exception as e:
            print(f"Error processing image: {e}")

    img_y -= + 50  # Space before class-wise count

    # **Class-Wise Image Count**
    if img_y < 100:  # Ensure enough space for class-wise count
        c.showPage()  # New page
        img_y = height - 100

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, img_y, "Class-Wise Image Count")
    img_y -= 20

    c.setFont("Helvetica", 12)
    for cls, count in class_counts.items():
        c.drawCentredString(width / 2, img_y, f"{cls}: {count} images")
        img_y -= 15

    # **Footer**
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 30, "© 2025 Tomato Leaf Disease Detector. All Rights Reserved.")

    # **Save and Return PDF**
    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="multiple_detection_report.pdf", mimetype="application/pdf")

@app.route('/necessary_precautions')
def necessary_precautions():
    return render_template('necessary_precautions.html')

if __name__ == '__main__':
    app.run(debug=True)