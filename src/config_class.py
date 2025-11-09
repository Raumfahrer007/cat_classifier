from dataclasses import dataclass, fields

@dataclass
class ConfigClass:
    """Base configuration class with common functionality"""
    
    def update(self, **kwargs):
        """Safely update configuration parameters"""
        valid_fields = {f.name for f in fields(self)}
        
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config field: {key}")
        
        print(f"Updated: {kwargs}")
        return self  # For method chaining
    
    def reset(self):
        """Reset to default values"""
        defaults = self.__class__()
        for field in fields(self):
            setattr(self, field.name, getattr(defaults, field.name))
        return self
    
    def copy(self):
        """Create a copy for experimentation"""
        return self.__class__(**self.__dict__)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = [f"{self.__class__.__name__}:"]
        for field in fields(self):
            value = getattr(self, field.name)
            lines.append(f"  {field.name}: {value}")
        return "\n".join(lines)