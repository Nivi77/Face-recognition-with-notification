A real-time face recognition system using DeepFace (SFace model) and OpenCV. 
The system generates face embeddings from known images, stores them in a NumPy database, and performs live recognition through a webcam.

Project structure
project-folder/
│
├── build_database.py
├── fast_recognition.py
├── knownfaces/
│     ├── person1.jpg
│     ├── person2.jpg
│
├── face_store/
│     ├── names.npy
│     └── embeddings.npy
│
├── requirements.txt
└── README.md

-------------
Step 1
Place images inside:
knownfaces/

Example:
knownfaces/
    niveditha.jpg
    arun.jpg
Each filename becomes the person name.
------------
Step 2 — Build Face Database
Run:
python build_database.py

This generates:
face_store/names.npy
face_store/embeddings.npy
-------------
 Step 3 — Start Real-Time Recognition
python tr1.py

Press:
------------
q → Quit camera

------------------------------------------------
Email Alerts
When a known face is detected:
An email alert is sent
Includes timestamp and confidence score
Make sure to: Use an App Password (Gmail), Update sender & receiver emails in the script

GPU Support (Optional)
If CUDA is installed:
TensorFlow automatically uses GPU
To check:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
