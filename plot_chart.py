from project_utilities import data_file_plotter
import sys

print('Starting')
data_file_plotter('.', filename=sys.argv[1], title=sys.argv[2], axis=(1, 85, .7, 1))
print('Done')

