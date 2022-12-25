# yolov5-api
## malsk4a_qappair_vb2011

![malsk4a_qappair_vb2011](https://user-images.githubusercontent.com/22097346/209459735-7b8a3dd7-e53d-4e83-8a46-e1eeb957d323.jpeg)

# API POST with Postman
<img width="1248" alt="Screen Shot 2565-12-25 at 14 06 19" src="https://user-images.githubusercontent.com/22097346/209459661-6aa6f709-b322-47ea-96cc-5107f1d4c311.png">



# Json Result 

```json
{
  "message": {
    "epoch_time": 1671951699.1293957,
    "error": false,
    "version": "1.2.0",
    "processing_time": [
      {
        "yolov5_process": 0.02654123306274414
      }
    ],
    "results": [
      {
        "confidence": 0.9289113283157349,
        "label": "car",
        "points": [
          539,
          246.5,
          304,
          251
        ],
        "type": "rectangle"
      },
      {
        "confidence": 0.9240482449531555,
        "label": "car",
        "points": [
          169.5,
          234,
          335,
          234
        ],
        "type": "rectangle"
      },
      {
        "confidence": 0.7312052845954895,
        "label": "car",
        "points": [
          35.5,
          195,
          71,
          124
        ],
        "type": "rectangle"
      }
    ]
  }
}
```

# Image Result

![response](https://user-images.githubusercontent.com/22097346/209459767-2d808241-869f-4a09-9e3f-0b531d3a4ae7.jpeg)
