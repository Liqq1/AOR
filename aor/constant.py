import os

DEFAULT_BOC_TOKEN = "["  # begin of coordinates
DEFAULT_EOC_TOKEN = "]"  # end of coordinates
DEFAULT_GRD_TOKEN = "<grounding>"
COT_ACTIVATION = "Answer the question and include the reasoning proess. Locate key objects and provide bounding boxes in your thoughts."
COT_ACTIVATION_TXT = "Answer the question and include the reasoning proess."
STR_BOX_TEMPLATE = ", ".join(["{" + ":.{}f".format(2) + "}"] * 4)
