from ptych import PtychStudy, CaptureRegion
from ptych.preview import show_study_capture, show_object_preview

study = PtychStudy.load("malaria-test")
region = CaptureRegion.centered_square(
    width=study.manifest.capture_dimensions.width,
    height=study.manifest.capture_dimensions.height,
    size=256,
)

show_study_capture(study, 0, capture_region=region)
show_object_preview("results/reconstruction_usaf_test/stitched_object.npy")
