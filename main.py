import cv2

# Load the video file or capture device (0 for the default camera)
video = cv2.VideoCapture("input/video_1.mp4")

# Initialize the background subtractor (use MOG2 or KNN for better results)
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # Break the loop if no frame is returned (end of video)
    if not ret:
        break

    # Apply background subtraction to get the foreground mask
    fg_mask = background_subtractor.apply(frame)

    # Remove noise using morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, None)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, None)

    # Find contours of the moving objects in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each detected contour
    for contour in contours:
        # Ignore small contours to reduce false detections
        if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
            # Get bounding box coordinates for each detected moving object
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected moving objects
    cv2.imshow("Moving Object Tracking", frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
