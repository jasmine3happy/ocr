import requests

PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'
def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        print(r["predictions"])
    # Otherwise, the request failed.
    else:
        print('Request failed')

predict_result('/Users/yunyubai/Downloads/U_disk/ocr/ctpn/datasetss/testing_data/images/6.jpg')