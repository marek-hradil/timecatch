## Introduction

We are conducting a study about how AI systems reason about the order of events in videos. Concretely: if you show AI a sequence of images from a video, can it tell if the images are in the right order? To answer this, we also need to know how well humans perform the same task.

You will be given series of four to eight **snapshots** from a video. The **goal** is to **evaluate** how well humans can detect if a swap happened in the sequence. This data will then be used to compare against how well AI performs. No other data than the annotations is collected (no personal data, location data, etc.).

## Your Task

You will interact with a dataset of simple 2D shapes (circles, squares, triangles in different colors and sizes) moving, colliding, rolling down ramps, and sometimes falling into a basket. Play the clip below to get a feel for how the dataset looks — since AI models perceive video as a series of still images rather than motion, we will work with individual snapshots rather than the video itself.

![Example clip from the dataset](https://storage.googleapis.com/lamp-annotation-datasets/introductions/craft-example.mp4)

You will be given a sequence where two **consecutive** images **are swapped**. Inspect the sequence and use the buttons to select **which two images** are swapped.

![Example of the user interface](https://storage.googleapis.com/lamp-annotation-datasets/introductions/craft-localize.png)

**Answer:** Frames 0 and 1.

**Tip:** Click any image to open a full-size viewer and use the **← →** arrow keys (or the on-screen buttons) to flip between frames one at a time — this makes it much easier to spot the swapped pair.

![Close-up of the image viewer](https://storage.googleapis.com/lamp-annotation-datasets/introductions/craft-localize-closeup.png)

The full workflow looks like this:

![Full workflow video](https://storage.googleapis.com/lamp-annotation-datasets/introductions/craft-localize-video.mp4)

The whole process should take around 15 minutes across 15 sequences. Thank you for your participation!
