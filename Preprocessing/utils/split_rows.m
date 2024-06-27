function cellArray = split_rows(matrix)
% Splits the matrix to one cell per row
cellArray = mat2cell(matrix, ones(1, height(matrix)));
end

