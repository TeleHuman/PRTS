from colorama import init, Fore, Style
from datetime import datetime

# Initialize colorama
init(autoreset=True)

class CustomPrinter:
    """Custom colored printer."""
    
    # Define message type configuration
    TYPE_CONFIG = {
        'normal': {
            'color': Fore.WHITE,
            'icon': '',
            'prefix': '',
            'style': Style.NORMAL
        },
        'info': {
            'color': Fore.BLUE,
            'icon': '',
            'prefix': 'INFO',
            'style': Style.NORMAL
        },
        'success': {
            'color': Fore.GREEN,
            'icon': '✅',
            'prefix': 'SUCCESS',
            'style': Style.BRIGHT
        },
        'warning': {
            'color': Fore.YELLOW,
            'icon': '',
            'prefix': 'WARNING',
            'style': Style.BRIGHT
        },
        'error': {
            'color': Fore.RED,
            'icon': '❌',
            'prefix': 'ERROR',
            'style': Style.BRIGHT
        },
        'fail': {
            'color': Fore.RED,
            'icon': '💥',
            'prefix': 'FAIL',
            'style': Style.BRIGHT
        },
        'debug': {
            'color': Fore.MAGENTA,
            'icon': '🐛',
            'prefix': 'DEBUG',
            'style': Style.NORMAL
        },
        'important': {
            'color': Fore.CYAN,
            'icon': '💡',
            'prefix': 'IMPORTANT',
            'style': Style.BRIGHT
        }
    }
    
    @classmethod
    def print(cls, message, msg_type='normal', show_time=True, show_icon=True, end='\n'):
        """
        Custom print function.
        
        Args:
            message: The message content to print
            msg_type: Message type ('normal', 'info', 'success', 'warning', 'error', 'fail', 'debug', 'important')
            show_time: Whether to display a timestamp
            show_icon: Whether to display the icon
            end: Line terminator
        """
        # Get configuration for the message type
        config = cls.TYPE_CONFIG.get(msg_type, cls.TYPE_CONFIG['normal'])
        
        # Build prefix parts
        prefix_parts = []
        
        # Add timestamp
        if show_time:
            timestamp = datetime.now().strftime('%H:%M:%S')
            prefix_parts.append(f"[{timestamp}]")
        
        # Add icon and prefix text
        icon_text = f"{config['icon']} " if show_icon else ""
        prefix_parts.append(f"{icon_text}{config['prefix']}")
        
        if config['prefix'] == '':
            full_message = message
        else:
            # Combine prefix parts
            prefix = " ".join(prefix_parts)
            
            # Construct full message
            full_message = f"{prefix}: {message}"
        
        # Apply color and style and print
        formatted_message = f"{config['style']}{config['color']}{full_message}"
        print(formatted_message, end=end)
    
    @classmethod
    def normal(cls, message, **kwargs):
        """Convenience: normal-level print."""
        cls.print(message, 'normal', **kwargs)

    @classmethod
    def info(cls, message, **kwargs):
        """Convenience: info-level print."""
        cls.print(message, 'info', **kwargs)
    
    @classmethod
    def success(cls, message, **kwargs):
        """Convenience: success-level print."""
        cls.print(message, 'success', **kwargs)
    
    @classmethod
    def warning(cls, message, **kwargs):
        """Convenience: warning-level print."""
        cls.print(message, 'warning', **kwargs)
    
    @classmethod
    def error(cls, message, **kwargs):
        """Convenience: error-level print."""
        cls.print(message, 'error', **kwargs)
    
    @classmethod
    def fail(cls, message, **kwargs):
        """Convenience: fail-level print."""
        cls.print(message, 'fail', **kwargs)
    
    @classmethod
    def debug(cls, message, **kwargs):
        """Convenience: debug-level print."""
        cls.print(message, 'debug', **kwargs)
    
    @classmethod
    def important(cls, message, **kwargs):
        """Convenience: important-level print."""
        cls.print(message, 'important', **kwargs)

# Create convenient global functions
def cprint(message, msg_type='normal', **kwargs):
    CustomPrinter.print(message, msg_type, **kwargs)

def normal(message, **kwargs):
    CustomPrinter.normal(message, **kwargs)

def info(message, **kwargs):
    CustomPrinter.info(message, **kwargs)

def success(message, **kwargs):
    CustomPrinter.success(message, **kwargs)

def warning(message, **kwargs):
    CustomPrinter.warning(message, **kwargs)

def error(message, **kwargs):
    CustomPrinter.error(message, **kwargs)

def fail(message, **kwargs):
    CustomPrinter.fail(message, **kwargs)

def debug(message, **kwargs):
    CustomPrinter.debug(message, **kwargs)

def important(message, **kwargs):
    CustomPrinter.important(message, **kwargs)
