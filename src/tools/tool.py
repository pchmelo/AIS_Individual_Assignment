

class Tool:
    def __init__(self, name, function, description, parameters):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters

        self.dict_description = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


    