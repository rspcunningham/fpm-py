from ptych import PtychStudy, CaptureRegion, show_study_capture

study = PtychStudy.load("malaria-test")
region = CaptureRegion.centered_square(
    width=study.manifest.capture_dimensions.width,
    height=study.manifest.capture_dimensions.height,
    size=256,
)

show_study_capture(study, 0, capture_region=region)
