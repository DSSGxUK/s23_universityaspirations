def format_school(school_name):
    """
    Formats the 'school' column in the pupil characteristics and destinations datasets

    Args:
        school_name: A value from n a pandas series

    Returns:
        A formated value from a pandas series
    """
    stopwords = [
        "academies",
        "academy",
        "and",
        "(boys)",
        "boys",
        "boys'",
        "buccleuch",
        "cc",
        "city",
        "college",
        "combined",
        "community",
        "for",
        "form",
        "(girls)",
        "girls",
        "girls'",
        "girls'",
        "grammar",
        "high",
        "infant",
        "junior",
        "nursery",
        "preparatory",
        "primary",
        "rother",
        "school",
        "science",
        "sixth",
        "technology",
        "the",
        "united",
    ]

    if isinstance(school_name, str):
        school_words = [x.lower() for x in school_name.split()]
        result = [word for word in school_words if word not in stopwords]
        new_school_name = " ".join(result)
        if new_school_name == "whgs":
            new_school_name = "william hulme's"
        if new_school_name == "tta":
            new_school_name = "totteridge"
        if new_school_name == "sheffieldsprings":
            new_school_name = "sheffield springs"
        if new_school_name == "lambeth":
            new_school_name = "elms"
        if new_school_name == "carter":
            new_school_name = "cornerstone"
        if new_school_name == "harrop fold":
            new_school_name = "lowry"
        if new_school_name == "glenmoor":
            new_school_name = "glenmoor & winton"
        if new_school_name == "winton":
            new_school_name = "glenmoor & winton"
        if new_school_name == "swindon - alton close":
            new_school_name = "swindon"
        if new_school_name == "swindon - beech avenue":
            new_school_name = "swindon"

    else:
        new_school_name = school_name

    return new_school_name
