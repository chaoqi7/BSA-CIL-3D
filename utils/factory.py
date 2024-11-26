def get_model(model_name, args, model_config = None):
    name = model_name.lower()
    if name == "ease":
        from models.ease import Learner
    else:
        assert 0
    
    return Learner(args, model_config = model_config)
