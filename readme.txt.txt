QureMe/
│
├── data/                     # For storing your MRI dataset
│   ├── Tumor/                # Images with tumors
│   └── No Tumor/             # Images without tumors
│
├── model/                    # Directory for storing trained models
│   └── brain_tumor_model.h5  # The final saved model (created after training)
│
├── 
│
├── scripts/                  # Python scripts for data processing and model training
│   └── preprocess.py         # Data preprocessing script
│   └── train_model.py        # Script to train the model
│   └── predict.py            # Python scripts for data training and preprocessing
│]
├── app/                      # Web application directory
│   └── app.py                # Flask app for serving the model
│   └── static  	      # Uploads images
│   └── UI	 	      # HTML pages
│   └── Uploads 	      # Saves uploaded images 
│
├── requirements.txt          # List of required Python packages
└── README.md                 # Project description