from ptych import PtychStudy, CaptureRegion
from ptych.preview import render_study_capture_preview_png

study = PtychStudy.load("usaf-test")
region = CaptureRegion.centered_square(
    width=study.manifest.capture_dimensions.width,
    height=study.manifest.capture_dimensions.height,
    size=256,
)

#show_study_capture(study, 0, capture_region=region)
#show_object_preview("results/reconstruction_usaf_test/stitched_object.npy")

render_study_capture_preview_png(study, 0, region, path="./results/usaf-test-2/capture_0.png")
