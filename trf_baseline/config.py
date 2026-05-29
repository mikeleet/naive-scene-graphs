# vg150 — 150 most frequent object classes, 50 most frequent predicates
# from danfeiX/scene-graph-TF-release and neural-motifs

OBJECT_CLASSES: list[str]
PREDICATES: list[str]

OBJECT_CLASSES = [
    "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear",
    "bed", "bench", "bike", "bird", "board", "boat", "book", "boot",
    "bottle", "bowl", "box", "boy", "branch", "building", "bus",
    "cap", "car", "cat", "chair", "child", "clock", "coat", "counter",
    "cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear",
    "elephant", "engine", "eye", "face", "fence", "finger", "flag", "flower",
    "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy",
    "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse",
    "house", "jacket", "jeans", "kid", "kite", "lady", "lamp", "laptop",
    "leaf", "leg", "letter", "light", "logo", "man", "men", "mirror",
    "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange",
    "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza",
    "plane", "plant", "plate", "player", "pole", "post", "pot", "racket",
    "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf",
    "shirt", "shoe", "shorts", "sidewalk", "sign", "sink", "skateboard",
    "ski", "skier", "sneaker", "snow", "sock", "stand", "street",
    "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel",
    "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase",
    "vegetable", "vehicle", "wave", "wheel", "window", "windshield", "wing",
    "wire", "woman", "zebra",
]

assert len(OBJECT_CLASSES) == 150

PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in",
    "covering", "eating", "flying in", "from", "growing on",
    "hanging from", "has", "holding", "in", "in front of", "laying on",
    "looking at", "lying on", "made of", "mounted on", "near", "of",
    "on", "on back of", "over", "painted on", "parked on", "part of",
    "playing", "riding", "says", "sitting on", "skating on", "skiing on",
    "standing on", "surfing on", "to", "under", "using", "walking in",
    "walking on", "watching", "wearing",
]

assert len(PREDICATES) == 50

OBJ2IDX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
PRED2IDX = {name: i for i, name in enumerate(PREDICATES)}

# number of categories per feature for CategoricalNB min_categories
FEATURE_N_CATEGORIES = [150, 150, 8, 8, 19, 10, 10]
