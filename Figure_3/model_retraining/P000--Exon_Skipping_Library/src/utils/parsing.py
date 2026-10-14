
def bool_map(string):
	"""
	maps 'true' and 'false' string booleans into true booleans
	"""

	if isinstance(string, str):
		string = string.lower()
		if string == "false":
			return False
		elif string == "true":
			return True
		elif string == "0":
			return False
		elif string == "1":
			return True
	elif isinstance(string, int):
		return string == 1
	elif isinstance(string, bool):
		return string


class ErrorWrapper():
    def __init__(self, opts, max_printout_size = 100):
        self.opts = opts
        self.missing_keys = []
        self.max_printout_size = max_printout_size
    def __getitem__(self, item):
        if item in self.opts:
            return self.opts[item]
        else:
            self.missing_keys.append(item)
            return "0"
    def get(self, item, default):
        return self.opts.get(item, default)
    def check_inputs(self):
        if self.missing_keys:
            raise KeyError(
                f"Missing required arguments: "
                f"{', '.join(self.missing_keys)}"
            )
        return 
    def __str__(self):
        out_str = []
        out_str.append("--- Received Inputs ---")
        max_key_len = 0
        for key in self.opts:
            max_key_len = max(max_key_len, len(key))
        for key in self.opts:
            val = self.opts[key]
            if len(str(self.opts[key])) > self.max_printout_size:
                val = str(self.opts[key])[0:self.max_printout_size] + " ..."
            out_str.append(f"{key.ljust(max_key_len)} {val}")
        return "\n".join(out_str)
    def __contains__(self, item):
        return item in self.opts
    def __len__(self):
        return len(self.opts)


class ParserHelper():
    def __init__(self, datatype):
        if datatype == "bool":
            self.map_func = bool_map
        elif datatype == "int":
            self.map_func = int
        elif datatype == "float":
            self.map_func = float
        else:
            self.map_func = str
    def __getitem__(self, item):
        return self.map_func(item)

 


if __name__ == "__main__":
    opts = {"--animal":"cow", "--food":"beef"*88, "--house":"yes"}
    opts = ErrorWrapper(opts)
    print(opts)
    pokemon = opts.get("--pokemon", "Psyduck")
    opts.check_inputs()
