# Make file for easier to start the conversion container
make_extract_container:
				docker build -t extract_container .; \

start_extract_data:
	docker run -it --rm \
	-v bag_files:/workspace/bag_files \
	-v converted:/workspace/converted \
	extract_container ; \