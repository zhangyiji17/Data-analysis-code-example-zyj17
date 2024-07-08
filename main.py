# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
data2 = b"\xFE\x0F\x00\x00\x00\x1f\x04\x91\x8A\x10\x00\xf2\x46"

def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte  # 异或运算
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1

    crc_low = crc & 0xFF
    crc_high = (crc >> 8) & 0xFF
    print(crc_low.to_bytes(1, byteorder='little'), crc_high.to_bytes(1, byteorder='little'))
    result = data + crc_low.to_bytes(1, byteorder='little') + crc_high.to_bytes(1, byteorder='little')
    if result == data2:
        print("True")
    return result


def crc16_modbus(data: bytes) -> bytes:
    """
    Calculate CRC-16 (Modbus) for the given data and append it to the data.

    Parameters:
    data (bytes): The Modbus data to calculate the CRC-16 for.

    Returns:
    bytes: The original data with the CRC-16 appended.
    """
    crc = 0xFFFF
    for byte in data:
        cur_byte = 0xFF & byte
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
            cur_byte = cur_byte >> 1

            # Append CRC to the data (low byte first, high byte second)
    crc_bytes = crc.to_bytes(2, byteorder='little')
    print(crc_bytes)
    result = data+crc_bytes
    print(type(result))
    return data + crc_bytes


# 示例用法
command = b'\x01\x03\x00\x00\x00\x01'  # 假设这是我们要发送的 Modbus 请求
full_command_with_crc = crc16_modbus(command)
#print(full_command_with_crc.hex())  # 打印包含CRC的完整命令的16进制表示
data = b'\xFE\x0F\x00\x00\x00\x20\x04\xFF\xFF\xFF\xFF'
data0 = b"\xFE\x0F\x00\x00\x00\x20\x04\x00\x00\x00\x00"
data1 = b"\xFE\x0F\x00\x00\x00\x1f\x04\x91\x8A\x10\x00"

def set_bit_at_index(indices):
    num = int(0)
    for i in indices:
        num |= (1 << (i-1))
    num = format(num, '08X')
    split_num = [num[i:i+2] for i in range(0, len(num), 2)]
    io = split_num[::-1]
    crc_input = "FE0F0000001f04" + ''.join(io)
    crc_input = [crc_input[j:j+2] for j in range(0, len(crc_input), 2)]
    crc_input = [int(n, 16) for n in crc_input if n != '']
    crc_input = bytes(crc_input)
    result = crc16(crc_input)
    return result



print(set_bit_at_index([1, 5, 8, 10, 21, 16, 12]))
#print(crc16(data1))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
