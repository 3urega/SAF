write_file = open("data/3274-Device-Data-Fix.csv", "w")

with open("data/3274-Device Data.csv", "r") as f:
    header = next(f)
    write_file.write(header.replace("\"", ""))
    for line in f:
        line_rep = line.replace("\"", "")
        day, date, data = line_rep.split(",", 2)
        date = "\"" + day + date + "\""
        data = data.replace("\"", "")
        fixed_line = date + "," + data
        write_file.write(fixed_line)