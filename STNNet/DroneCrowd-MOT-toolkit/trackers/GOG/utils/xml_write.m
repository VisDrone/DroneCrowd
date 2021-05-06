function DOMnode = xml_write(filename, tree, RootName, Pref)
%XML_WRITE  Writes Matlab data structures to XML file
%
% DESCRIPTION
% xml_write( filename, tree) Converts Matlab data structure 'tree' containing
% cells, structs, numbers and strings to Document Object Model (DOM) node
% tree, then saves it to XML file 'filename' using Matlab's xmlwrite
% function. Optionally one can also use alternative version of xmlwrite
% function which directly calls JAVA functions for XML writing without
% MATLAB middleware. This function is provided as a patch to existing
% bugs in xmlwrite (in R2006b).
%
% xml_write(filename, tree, RootName, Pref) allows you to specify
% additional preferences about file format
%
% DOMnode = xml_write([], tree) same as above except that DOM node is
% not saved to the file but returned.
%
% INPUT
%   filename     file name
%   tree         Matlab structure tree to store in xml file.
%   RootName     String with XML tag name used for root (top level) node
%                Optionally it can be a string cell array storing: Name of
%                root node, document "Processing Instructions" data and
%                document "comment" string
%   Pref         Other preferences:
%     Pref.ItemName - default 'item' -  name of a special tag used to
%                     itemize cell or struct arrays
%     Pref.XmlEngine - let you choose the XML engine. Currently default is
%       'Xerces', which is using directly the apache xerces java file.
%       Other option is 'Matlab' which uses MATLAB's xmlwrite and its
%       XMLUtils java file. Both options create identical results except in
%       case of CDATA sections where xmlwrite fails.
%     Pref.CellItem - default 'true' - allow cell arrays to use 'item'
%       notation. See below.
%    Pref.RootOnly - default true - output variable 'tree' corresponds to
%       xml file root element, otherwise it correspond to the whole file.
%     Pref.StructItem - default 'true' - allow arrays of structs to use
%       'item' notation. For example "Pref.StructItem = true" gives:
%         <a>
%           <b>
%             <item> ... <\item>
%             <item> ... <\item>
%           <\b>
%         <\a>
%       while "Pref.StructItem = false" gives:
%         <a>
%           <b> ... <\b>
%           <b> ... <\b>
%         <\a>
%
%
% Several special xml node types can be created if special tags are used
% for field names of 'tree' nodes:
%  - node.CONTENT - stores data section of the node if other fields
%    (usually ATTRIBUTE are present. Usually data section is stored
%    directly in 'node'.
%  - node.ATTRIBUTE.name - stores node's attribute called 'name'.
%  - node.COMMENT - create comment child node from the string. For global
%    comments see "RootName" input variable.
%  - node.PROCESSING_INSTRUCTIONS - create "processing instruction" child
%    node from the string. For global "processing instructions" see
%    "RootName" input variable.
%  - node.CDATA_SECTION - stores node's CDATA section (string). Only works
%    if Pref.XmlEngine='Xerces'. For more info, see comments of F_xmlwrite.
%  - other special node types like: document fragment nodes, document type
%    nodes, entity nodes and notation nodes are not being handled by
%    'xml_write' at the moment.
%
% OUTPUT
%   DOMnode      Document Object Model (DOM) node tree in the format
%                required as input to xmlwrite. (optional)
%
% EXAMPLES:
%   MyTree=[];
%   MyTree.MyNumber = 13;
%   MyTree.MyString = 'Hello World';
%   xml_write('test.xml', MyTree);
%   type('test.xml')
%   %See also xml_tutorial.m
%
% See also
%   xml_read, xmlread, xmlwrite
%
% Written by Jarek Tuszynski, SAIC, jaroslaw.w.tuszynski_at_saic.com

%% Check Matlab Version
v = ver('MATLAB');
v = str2double(regexp(v.Version, '\d.\d','match','once'));
if (v<7)
  error('Your MATLAB version is too old. You need version 7.0 or newer.');
end

%% default preferences
DPref.TableName  = {'tr','td'}; % name of a special tags used to itemize 2D cell arrays
DPref.ItemName   = 'item'; % name of a special tag used to itemize 1D cell arrays
DPref.StructItem = true;  % allow arrays of structs to use 'item' notation
DPref.CellItem   = true;  % allow cell arrays to use 'item' notation
DPref.StructTable= 'Html';
DPref.CellTable  = 'Html';
DPref.XmlEngine  = 'Matlab';  % use matlab provided XMLUtils
%DPref.XmlEngine  = 'Xerces';  % use Xerces xml generator directly
DPref.PreserveSpace = false; % Preserve or delete spaces at the beggining and the end of stings?
RootOnly         = true;  % Input is root node only
GlobalProcInst = [];
GlobalComment  = [];
GlobalDocType  = [];

%% read user preferences
if (nargin>3)
  if (isfield(Pref, 'TableName' )),  DPref.TableName  = Pref.TableName; end
  if (isfield(Pref, 'ItemName'  )), DPref.ItemName   = Pref.ItemName;   end
  if (isfield(Pref, 'StructItem')), DPref.StructItem = Pref.StructItem; end
  if (isfield(Pref, 'CellItem'  )), DPref.CellItem   = Pref.CellItem;   end
  if (isfield(Pref, 'CellTable')),   DPref.CellTable  = Pref.CellTable; end
  if (isfield(Pref, 'StructTable')), DPref.StructTable= Pref.StructTable; end
  if (isfield(Pref, 'XmlEngine' )), DPref.XmlEngine  = Pref.XmlEngine;  end
  if (isfield(Pref, 'RootOnly'  )), RootOnly         = Pref.RootOnly;   end
  if (isfield(Pref, 'PreserveSpace')), DPref.PreserveSpace = Pref.PreserveSpace; end
end
if (nargin<3 || isempty(RootName)), RootName=inputname(2); end
if (isempty(RootName)), RootName='ROOT'; end
if (iscell(RootName)) % RootName also stores global text node data
  rName = RootName;
  RootName = char(rName{1});
  if (length(rName)>1), GlobalProcInst = char(rName{2}); end
  if (length(rName)>2), GlobalComment  = char(rName{3}); end
  if (length(rName)>3), GlobalDocType  = char(rName{4}); end
end
if(~RootOnly && isstruct(tree))  % if struct than deal with each field separatly
  fields = fieldnames(tree);
  for i=1:length(fields)
    field = fields{i};
    x = tree(1).(field);
    if (strcmp(field, 'COMMENT'))
      GlobalComment = x;
    elseif (strcmp(field, 'PROCESSING_INSTRUCTION'))
      GlobalProcInst = x;
    elseif (strcmp(field, 'DOCUMENT_TYPE'))
      GlobalDocType = x;
    else
      RootName = field;
      t = x;
    end
  end
  tree = t;
end

%% Initialize jave object that will store xml data structure
RootName = varName2str(RootName);
if (~isempty(GlobalDocType))
  %   n = strfind(GlobalDocType, ' ');
  %   if (~isempty(n))
  %     dtype = com.mathworks.xml.XMLUtils.createDocumentType(GlobalDocType);
  %   end
  %   DOMnode = com.mathworks.xml.XMLUtils.createDocument(RootName, dtype);
  warning('xml_io_tools:write:docType', ...
    'DOCUMENT_TYPE node was encountered which is not supported yet. Ignoring.');
end
DOMnode = com.mathworks.xml.XMLUtils.createDocument(RootName);


%% Use recursive function to convert matlab data structure to XML
root = DOMnode.getDocumentElement;
struct2DOMnode(DOMnode, root, tree, DPref.ItemName, DPref);

%% Remove the only child of the root node
root   = DOMnode.getDocumentElement;
Child  = root.getChildNodes; % create array of children nodes
nChild = Child.getLength;    % number of children
if (nChild==1)
  node = root.removeChild(root.getFirstChild);
  while(node.hasChildNodes)
    root.appendChild(node.removeChild(node.getFirstChild));
  end
  while(node.hasAttributes)            % copy all attributes
    root.setAttributeNode(node.removeAttributeNode(node.getAttributes.item(0)));
  end
end

%% Save exotic Global nodes
if (~isempty(GlobalComment))
  DOMnode.insertBefore(DOMnode.createComment(GlobalComment), DOMnode.getFirstChild());
end
if (~isempty(GlobalProcInst))
  n = strfind(GlobalProcInst, ' ');
  if (~isempty(n))
    proc = DOMnode.createProcessingInstruction(GlobalProcInst(1:(n(1)-1)),...
      GlobalProcInst((n(1)+1):end));
    DOMnode.insertBefore(proc, DOMnode.getFirstChild());
  end
end
% Not supported yet as the code below does not work
% if (~isempty(GlobalDocType))
%   n = strfind(GlobalDocType, ' ');
%   if (~isempty(n))
%     dtype = DOMnode.createDocumentType(GlobalDocType);
%     DOMnode.insertBefore(dtype, DOMnode.getFirstChild());
%   end
% end

%% save java DOM tree to XML file
if (~isempty(filename))
  if (strcmpi(DPref.XmlEngine, 'Xerces'))
    xmlwrite_xerces(filename, DOMnode);
  else
    xmlwrite(filename, DOMnode);
  end
end


%% =======================================================================
%  === struct2DOMnode Function ===========================================
%  =======================================================================
function [] = struct2DOMnode(xml, parent, s, TagName, Pref)
% struct2DOMnode is a recursive function that converts matlab's structs to
% DOM nodes.
% INPUTS:
%  xml - jave object that will store xml data structure
%  parent - parent DOM Element
%  s - Matlab data structure to save
%  TagName - name to be used in xml tags describing 's'
%  Pref - preferenced
% OUTPUT:
%  parent - modified 'parent'

% perform some conversions
if (ischar(s) && min(size(s))>1) % if 2D array of characters
  s=cellstr(s);                  % than convert to cell array
end
% if (strcmp(TagName, 'CONTENT'))
%   while (iscell(s) && length(s)==1), s = s{1}; end % unwrap cell arrays of length 1
% end
TagName  = varName2str(TagName);

%% == node is a 2D cell array ==
% convert to some other format prior to further processing
nDim = nnz(size(s)>1);  % is it a scalar, vector, 2D array, 3D cube, etc?
if (iscell(s) && nDim==2 && strcmpi(Pref.CellTable, 'Matlab'))
  s = var2str(s, Pref.PreserveSpace);
end
if (nDim==2 && (iscell  (s) && strcmpi(Pref.CellTable,   'Vector')) || ...
               (isstruct(s) && strcmpi(Pref.StructTable, 'Vector')))
  s = s(:);
end
if (nDim>2), s = s(:); end % can not handle this case well
nItem = numel(s);
nDim  = nnz(size(s)>1);  % is it a scalar, vector, 2D array, 3D cube, etc?

%% == node is a cell ==
if (iscell(s)) % if this is a cell or cell array
  if ((nDim==2 && strcmpi(Pref.CellTable,'Html')) || (nDim< 2 && Pref.CellItem))
    % if 2D array of cells than can use HTML-like notation or if 1D array
    % than can use item notation
    if (strcmp(TagName, 'CONTENT')) % CONTENT nodes already have <TagName> ... </TagName>
      array2DOMnode(xml, parent, s, Pref.ItemName, Pref ); % recursive call
    else
      node = xml.createElement(TagName);   % <TagName> ... </TagName>
      array2DOMnode(xml, node, s, Pref.ItemName, Pref ); % recursive call
      parent.appendChild(node);
    end
  else % use  <TagName>...<\TagName> <TagName>...<\TagName> notation
    array2DOMnode(xml, parent, s, TagName, Pref ); % recursive call
  end
%% == node is a struct ==
elseif (isstruct(s))  % if struct than deal with each field separatly
  if ((nDim==2 && strcmpi(Pref.StructTable,'Html')) || (nItem>1 && Pref.StructItem))
    % if 2D array of structs than can use HTML-like notation or
    % if 1D array of structs than can use 'items' notation
    node = xml.createElement(TagName);
    array2DOMnode(xml, node, s, Pref.ItemName, Pref ); % recursive call
    parent.appendChild(node);
  elseif (nItem>1) % use  <TagName>...<\TagName> <TagName>...<\TagName> notation
    array2DOMnode(xml, parent, s, TagName, Pref ); % recursive call
  else % otherwise save each struct separatelly
    fields = fieldnames(s);
    node = xml.createElement(TagName);
    for i=1:length(fields) % add field by field to the node
      field = fields{i};
      x = s.(field);
      switch field
        case {'COMMENT', 'CDATA_SECTION', 'PROCESSING_INSTRUCTION'}
          if iscellstr(x)  % cell array of strings -> add them one by one
            array2DOMnode(xml, node, x(:), field, Pref ); % recursive call will modify 'node'
          elseif ischar(x) % single string -> add it
            struct2DOMnode(xml, node, x, field, Pref ); % recursive call will modify 'node'
          else % not a string - Ignore
            warning('xml_io_tools:write:badSpecialNode', ...
             ['Struct field named ',field,' encountered which was not a string. Ignoring.']);
          end
        case 'ATTRIBUTE' % set attributes of the node
          if (isempty(x)), continue; end
          if (isstruct(x))
            attName = fieldnames(x);       % get names of all the attributes
            for k=1:length(attName)        % attach them to the node
              att = xml.createAttribute(varName2str(attName(k)));
              att.setValue(var2str(x.(attName{k}),Pref.PreserveSpace));
              node.setAttributeNode(att);
            end
          else
            warning('xml_io_tools:write:badAttribute', ...
              'Struct field named ATTRIBUTE encountered which was not a struct. Ignoring.');
          end
        otherwise                            % set children of the node
          struct2DOMnode(xml, node, x, field, Pref ); % recursive call will modify 'node'
      end
    end  % end for i=1:nFields
    parent.appendChild(node);
  end
%% == node is a leaf node ==
else  % if not a struct and not a cell than it is a leaf node
  switch TagName % different processing depending on desired type of the node
    case 'COMMENT'   % create comment node
      com = xml.createComment(s);
      parent.appendChild(com);
    case 'CDATA_SECTION' % create CDATA Section
      cdt = xml.createCDATASection(s);
      parent.appendChild(cdt);
    case 'PROCESSING_INSTRUCTION' % set attributes of the node
      OK = false;
      if (ischar(s))
        n = strfind(s, ' ');
        if (~isempty(n))
          proc = xml.createProcessingInstruction(s(1:(n(1)-1)),s((n(1)+1):end));
          parent.insertBefore(proc, parent.getFirstChild());
          OK = true;
        end
      end
      if (~OK)
        warning('xml_io_tools:write:badProcInst', ...
          ['Struct field named PROCESSING_INSTRUCTION need to be',...
          ' a string, for example: xml-stylesheet type="text/css" ', ...
          'href="myStyleSheet.css". Ignoring.']);
      end
    case 'CONTENT' % this is text part of already existing node
      txt  = xml.createTextNode(var2str(s, Pref.PreserveSpace)); % convert to text
      parent.appendChild(txt);
    otherwise      % I guess it is a regular text leaf node
      txt  = xml.createTextNode(var2str(s, Pref.PreserveSpace));
      node = xml.createElement(TagName);
      node.appendChild(txt);
      parent.appendChild(node);
  end
end % of struct2DOMnode function

%% =======================================================================
%  === array2DOMnode Function ============================================
%  =======================================================================
function [] = array2DOMnode(xml, parent, s, TagName, Pref)
% Deal with 1D and 2D arrays of cell or struct. Will modify 'parent'.
nDim = nnz(size(s)>1);  % is it a scalar, vector, 2D array, 3D cube, etc?
switch nDim
  case 2 % 2D array
    for r=1:size(s,1)
      subnode = xml.createElement(Pref.TableName{1});
      for c=1:size(s,2)
        v = s(r,c);
        if iscell(v), v = v{1}; end
        struct2DOMnode(xml, subnode, v, Pref.TableName{2}, Pref ); % recursive call
      end
      parent.appendChild(subnode);
    end
  case 1 %1D array
    for iItem=1:numel(s)
      v = s(iItem);
      if iscell(v), v = v{1}; end
      struct2DOMnode(xml, parent, v, TagName, Pref ); % recursive call
    end
  case 0 % scalar -> this case should never be called
    if ~isempty(s) 
      if iscell(s), s = s{1}; end
      struct2DOMnode(xml, parent, s, TagName, Pref );
    end
end

%% =======================================================================
%  === var2str Function ==================================================
%  =======================================================================
function str = var2str(object, PreserveSpace)
% convert matlab variables to a string
switch (1)
  case isempty(object)
    str = '';
  case (isnumeric(object) || islogical(object))
    if ndims(object)>2, object=object(:); end  % can't handle arrays with dimention > 2
    str=mat2str(object);           % convert matrix to a string
    % mark logical scalars with [] (logical arrays already have them) so the xml_read
    % recognizes them as MATLAB objects instead of strings. Same with sparse
    % matrices
    if ((islogical(object) && isscalar(object)) || issparse(object)),
      str = ['[' str ']'];
    end
    if (isinteger(object)),
      str = ['[', class(object), '(', str ')]'];
    end
  case iscell(object)
    if ndims(object)>2, object=object(:); end  % can't handle cell arrays with dimention > 2
    [nr nc] = size(object);
    obj2 = object;
    for i=1:length(object(:))
      str = var2str(object{i}, PreserveSpace);
      if (ischar(object{i})), object{i} = ['''' object{i} '''']; else object{i}=str; end
      obj2{i} = [object{i} ','];
    end
    for r = 1:nr, obj2{r,nc} = [object{r,nc} ';']; end
    obj2 = obj2.';
    str = ['{' obj2{:} '}'];
  case isstruct(object)
    str='';
    warning('xml_io_tools:write:var2str', ...
      'Struct was encountered where string was expected. Ignoring.');
  case isa(object, 'function_handle')
    str = ['[@' char(object) ']'];
  case ischar(object)
    str = object;
  otherwise
    str = char(object);
end

%% string clean-up
str=str(:); str=str.';            % make sure this is a row vector of char's
if (~isempty(str))
  str(str<32|str==127)=' ';       % convert no-printable characters to spaces
  if (~PreserveSpace)
    str = strtrim(str);             % remove spaces from begining and the end
    str = regexprep(str,'\s+',' '); % remove multiple spaces
  end
end

%% =======================================================================
%  === var2Namestr Function ==============================================
%  =======================================================================
function str = varName2str(str)
% convert matlab variable names to a sting
str = char(str);
p   = strfind(str,'0x');
if (~isempty(p))
  for i=1:length(p)
    before = str( p(i)+(0:3) );          % string to replace
    after  = char(hex2dec(before(3:4))); % string to replace with
    str = regexprep(str,before,after, 'once', 'ignorecase');
    p=p-3; % since 4 characters were replaced with one - compensate
  end
end
str = regexprep(str,'_COLON_',':', 'once', 'ignorecase');
str = regexprep(str,'_DASH_' ,'-', 'once', 'ignorecase');

