import struct

def reinterpret_int_as_float(integer_value):
    # Ensure that the integer is within the range of a 32-bit signed integer
    if not (-2**31 <= integer_value < 2**32):
        raise ValueError("Integer must be within the range of a 32-bit signed integer.")
    
    # Pack the integer as a 32-bit integer and unpack it as a 32-bit float
    packed_integer = struct.pack('I', integer_value)  # 'I' is for unsigned 32-bit integer
    float_value = struct.unpack('f', packed_integer)[0]  # 'f' is for 32-bit float
    
    return float_value